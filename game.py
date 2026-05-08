import pygame
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import time
import os
import json

# === PARAMÈTRES SIMULATION ===
SETTINGS = {
    "SCREEN_WIDTH": 1000,
    "SCREEN_HEIGHT": 600,
    "PIPE_WIDTH": 80,
    "PIPE_HEIGHT": 500,
    "PIPE_GAP": 175,
    "PIPE_VELOCITY": 2, 
    "BIRD_SIZE": 30, 
    "GRAVITY": 0.5,
    "FLAP_STRENGTH": -10,

    "HIDDEN_SIZE": 32,
    "POP_SIZE": 25,
    "GENERATIONS": 50,
    "MAX_STEPS": 5000,
    "MUTATION_STD": 0.05,
    "MUTATION_RATE": 0.2,

    "FPS": 60,
    "VISUAL_DELAY": 10
}

CURRENT_DIR = os.path.dirname(__file__)
ASSETS_PATH = os.path.join(CURRENT_DIR, "..", "..", "images")

# === CLASSES ===
class FlappyBirdGame:
    def __init__(self, render_mode="human"):
        if not pygame.get_init():
            pygame.init()

        self.settings = SETTINGS
        self.SCREEN_WIDTH = self.settings["SCREEN_WIDTH"]
        self.SCREEN_HEIGHT = self.settings["SCREEN_HEIGHT"]
        self.PIPE_WIDTH = self.settings["PIPE_WIDTH"]
        self.PIPE_HEIGHT = self.settings["PIPE_HEIGHT"]
        self.PIPE_GAP = self.settings["PIPE_GAP"]
        self.PIPE_VELOCITY = self.settings["PIPE_VELOCITY"]
        self.BIRD_SIZE = self.settings["BIRD_SIZE"]
        self.GRAVITY = self.settings["GRAVITY"]
        self.FLAP_STRENGTH = self.settings["FLAP_STRENGTH"]
        self.mg = 180

        self.render_mode = render_mode
        self.screen = None
        self.next_pipe_id = 0
        
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption("Flappy Bird")

            self.pipe_img = pygame.image.load(os.path.join(ASSETS_PATH, "pipe_img.png")).convert_alpha()
            self.pipe_img = pygame.transform.scale(self.pipe_img, (325, 600))
            self.pipe_img_flipped = pygame.transform.flip(self.pipe_img, False, True)

            self.bird_img = pygame.image.load(os.path.join(ASSETS_PATH, "bird.png")).convert_alpha()
            self.bird_img = pygame.transform.scale(self.bird_img, (self.BIRD_SIZE * 2, self.BIRD_SIZE * 1.5))

        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.bird_y = self.SCREEN_HEIGHT // 2
        self.bird_velocity = 0
        self.pipe_pairs = [self.create_pipe_pair()]
        self.score = 0
        self.passed_pipes = set()
        self.done = False
        return self.get_obs()

    def create_pipe_pair(self):
        gap_y = random.randint(50, self.SCREEN_HEIGHT - 50 - self.PIPE_GAP)
        top_pipe = pygame.Rect(self.SCREEN_WIDTH, 0, self.PIPE_WIDTH, gap_y)
        bottom_pipe = pygame.Rect(self.SCREEN_WIDTH, gap_y + self.PIPE_GAP, self.PIPE_WIDTH,
                                  self.SCREEN_HEIGHT - (gap_y + self.PIPE_GAP))
        pipe_id = self.next_pipe_id
        self.next_pipe_id += 1
        return {"id": pipe_id, "top": top_pipe, "bottom": bottom_pipe}

    def update_pipes(self, bird_rect, draw=False, fps_augmented=False):
        reward = 0
        new_pipe_needed = False
        self.pipe_vel = self.PIPE_VELOCITY * 2 if fps_augmented else self.PIPE_VELOCITY

        for i, pipe_pair in enumerate(self.pipe_pairs):
            top_pipe = pipe_pair["top"]
            bottom_pipe = pipe_pair["bottom"]
            pipe_id = pipe_pair["id"]

            top_pipe.x -= self.pipe_vel
            bottom_pipe.x -= self.pipe_vel

            if draw and self.screen:
                top_x = top_pipe.centerx - self.pipe_img.get_width() // 2 - 6
                top_y = top_pipe.bottom - self.pipe_img_flipped.get_height() + self.mg
                bottom_x = bottom_pipe.centerx - self.pipe_img.get_width() // 2 - 6
                bottom_y = bottom_pipe.top - self.mg

                self.screen.blit(self.pipe_img, (top_x, top_y))
                self.screen.blit(self.pipe_img_flipped, (bottom_x, bottom_y))

            if bird_rect.colliderect(top_pipe) or bird_rect.colliderect(bottom_pipe):
                self.done = True
                reward = -3

            if top_pipe.right < bird_rect.left:
                if pipe_id not in self.passed_pipes:
                    self.score += 1
                    reward += 5
                    self.passed_pipes.add(pipe_id)

            if top_pipe.centerx < self.SCREEN_WIDTH // 2 and i == len(self.pipe_pairs) - 1:
                new_pipe_needed = True

        if new_pipe_needed:
            self.pipe_pairs.append(self.create_pipe_pair())

        self.pipe_pairs = [pair for pair in self.pipe_pairs if pair["top"].right > 0]
        return reward

    def draw_pipes(self):
        if not self.screen:
            return

        for pipe_pair in self.pipe_pairs:
            top_pipe = pipe_pair["top"]
            bottom_pipe = pipe_pair["bottom"]
            top_x = top_pipe.centerx - self.pipe_img.get_width() // 2 - 6
            top_y = top_pipe.bottom - self.pipe_img_flipped.get_height() + self.mg
            bottom_x = bottom_pipe.centerx - self.pipe_img.get_width() // 2 - 6
            bottom_y = bottom_pipe.top - self.mg

            self.screen.blit(self.pipe_img, (top_x, top_y))
            self.screen.blit(self.pipe_img_flipped, (bottom_x, bottom_y))

    def get_obs(self):
        next_pipe = None
        for pipe_pair in self.pipe_pairs:
            top = pipe_pair["top"]
            bottom = pipe_pair["bottom"]
            if top.right >= 50:
                next_pipe = pipe_pair
                break

        if next_pipe:
            gap_top = next_pipe["top"].height
            gap_bottom = next_pipe["bottom"].top
            pipe_x_dist = next_pipe["top"].left - 50
            pipe_y_center = next_pipe["top"].height + self.PIPE_GAP / 2
        else:
            gap_top = 0
            gap_bottom = self.SCREEN_HEIGHT
            pipe_x_dist = 0
            pipe_y_center = self.SCREEN_HEIGHT / 2

        return [
            self.bird_y / self.SCREEN_HEIGHT,
            self.bird_velocity / 10.0,
            pipe_x_dist / self.SCREEN_WIDTH,
            (pipe_y_center - self.bird_y) / self.SCREEN_HEIGHT,
            gap_top / self.SCREEN_HEIGHT,
            gap_bottom / self.SCREEN_HEIGHT
        ]

    def step(self, action):
        if self.done:
            return self.get_obs(), 0, True, {}

        if action == 1:
            self.bird_velocity = self.FLAP_STRENGTH

        self.bird_velocity += self.GRAVITY
        self.bird_y += self.bird_velocity

        bird_rect = pygame.Rect(50, int(self.bird_y), self.BIRD_SIZE, self.BIRD_SIZE)
        if self.screen:
            self.screen.blit(self.bird_img, (bird_rect.x - 20, bird_rect.y - 20))
        reward = self.update_pipes(bird_rect)

        if self.bird_y > self.SCREEN_HEIGHT or self.bird_y < 0:
            self.done = True
            reward = -2

        return self.get_obs(), reward + 0.1, self.done, {}

    def run(self):
        running = True
        self.reset()
        pygame.time.wait(500)

        while running:
            self.clock.tick(SETTINGS["FPS"])
            action = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        action = 1

            if action == 1:
                self.bird_velocity = self.FLAP_STRENGTH

            self.bird_velocity += self.GRAVITY
            self.bird_y += self.bird_velocity
            bird_rect = pygame.Rect(50, int(self.bird_y), self.BIRD_SIZE, self.BIRD_SIZE)

            reward = self.update_pipes(bird_rect)

            if self.bird_y < 0 or self.bird_y > self.SCREEN_HEIGHT:
                self.done = True

            self.screen.fill((135, 206, 235))
            self.draw_pipes()
            angle = -self.bird_velocity * 3
            rotated_bird = pygame.transform.rotate(self.bird_img, angle)
            self.screen.blit(rotated_bird, (bird_rect.x - 20, bird_rect.y - 10))

            font = pygame.font.Font(None, 36)
            text = font.render(f"Score: {self.score}", True, (0, 0, 0))
            self.screen.blit(text, (10, 10))
            pygame.display.flip()

            if self.done:
                pygame.time.wait(1000)
                self.reset()
                self.done = False
