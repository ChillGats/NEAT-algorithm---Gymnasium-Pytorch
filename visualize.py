import pygame
import random
import numpy as np
import torch
from game import FlappyBirdGame, SETTINGS
from model import NeuralNet, DEVICE
import train

load_generation_models = train.load_generation_models

def visualize_generation():
    generation = int(input("Quelle génération visualiser ? "))
    models, seed = load_generation_models(generation)
    if models is None: return

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    base_game = FlappyBirdGame(render_mode="human")
    game_clock = pygame.time.Clock()

    birds = []
    for model in models:
        bird = {
            "y": base_game.SCREEN_HEIGHT // 2,
            "velocity": 0,
            "alive": True,
            "model": model,
            "color": (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)),
            "score": 0,
            "passed_pipes": set()
        }
        birds.append(bird)

    steps = 0
    done = False
    speed_multiplier = 1
    
    while not done:
        game_clock.tick(SETTINGS["FPS"])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    speed_multiplier = max(1, speed_multiplier // 2)
                elif event.key == pygame.K_RIGHT:
                    speed_multiplier = min(960, speed_multiplier * 2)
                elif event.key == pygame.K_UP:
                    speed_multiplier = min(960, speed_multiplier + 1)
                elif event.key == pygame.K_DOWN:
                    speed_multiplier = max(1, speed_multiplier - 1)

        base_game.screen.fill((135, 206, 235))
        dummy_rect = pygame.Rect(50, 0, base_game.BIRD_SIZE, base_game.BIRD_SIZE)
        base_game.update_pipes(dummy_rect, draw=True, fps_augmented=True)

        all_dead = True
        for bird in birds:
            if not bird["alive"]: continue

            next_pipe = base_game.pipe_pairs[0]
            obs = [
                bird["y"] / base_game.SCREEN_HEIGHT,
                bird["velocity"] / 10.0,
                next_pipe["top"].left / base_game.SCREEN_WIDTH,
                ((next_pipe["top"].height + base_game.PIPE_GAP / 2) - bird["y"]) / base_game.SCREEN_HEIGHT,
                next_pipe["top"].height / base_game.SCREEN_HEIGHT,
                next_pipe["bottom"].top / base_game.SCREEN_HEIGHT
            ]

            x = torch.tensor(obs, dtype=torch.float32).to(DEVICE)
            output = bird["model"](x).item()
            action = 1 if output > 0.5 else 0

            if action == 1: bird["velocity"] = base_game.FLAP_STRENGTH
            bird["velocity"] += base_game.GRAVITY
            bird["y"] += bird["velocity"]

            bird_rect = pygame.Rect(50, int(bird["y"]), base_game.BIRD_SIZE, base_game.BIRD_SIZE)

            angle = -base_game.bird_velocity * 3
            rotated_bird = pygame.transform.rotate(base_game.bird_img, angle)
            base_game.screen.blit(rotated_bird, (bird_rect.x - 20, bird_rect.y - 10))

            # Check collisions and score for all pipes
            for pipe_pair in base_game.pipe_pairs:
                top_pipe = pipe_pair["top"]
                bottom_pipe = pipe_pair["bottom"]
                pipe_id = pipe_pair["id"]

                if bird_rect.colliderect(top_pipe) or bird_rect.colliderect(bottom_pipe):
                    bird["alive"] = False
                    break
                if bird["y"] < 0 or bird["y"] > base_game.SCREEN_HEIGHT:
                    bird["alive"] = False
                    break
                if top_pipe.right < bird_rect.left and pipe_id not in bird["passed_pipes"]:
                    bird["score"] += 1
                    bird["passed_pipes"].add(pipe_id)

            if not bird["alive"]:
                continue

            all_dead = False

        # Display speed multiplier
        font = pygame.font.Font(None, 28)
        speed_text = font.render(f"Speed: {speed_multiplier}x", True, (255, 255, 255))
        pygame.draw.rect(base_game.screen, (0, 0, 0), (5, 5, 150, 30))
        base_game.screen.blit(speed_text, (10, 10))
        
        pygame.display.flip()
        steps += 1
        if all_dead: done = True

    print("✅ Visualisation terminée.")
    random.seed()
    pygame.time.wait(1500)

def play_best_ai():
    if train.best_model_ever is None:
        print("❌ Aucun modèle chargé ou entraîné.")
        return
    
    game = FlappyBirdGame(render_mode="human")
    game_clock = pygame.time.Clock()
    obs = game.reset()
    running = True
    speed_multiplier = 1
    
    while running:
        game_clock.tick(SETTINGS["FPS"])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                running = False
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    speed_multiplier = max(1, speed_multiplier // 2)
                elif event.key == pygame.K_RIGHT:
                    speed_multiplier = min(960, speed_multiplier * 2)
                elif event.key == pygame.K_UP:
                    speed_multiplier = min(960, speed_multiplier + 1)
                elif event.key == pygame.K_DOWN:
                    speed_multiplier = max(1, speed_multiplier - 1)

        done = False
        for _ in range(speed_multiplier):
            x = torch.tensor(np.array(obs), dtype=torch.float32).to(DEVICE)
            output = train.best_model_ever(x).item()
            action = 1 if output > 0.5 else 0
            obs, reward, done, _ = game.step(action)
            if done:
                break

        game.screen.fill((135, 206, 235))
        game.draw_pipes()
        bird_rect = pygame.Rect(50, int(game.bird_y), game.BIRD_SIZE, game.BIRD_SIZE)

        angle = -game.bird_velocity * 3
        rotated_bird = pygame.transform.rotate(game.bird_img, angle)
        game.screen.blit(rotated_bird, (bird_rect.x -20 , bird_rect.y -10))

        font = pygame.font.Font(None, 36)
        text = font.render(f"Score: {game.score}", True, (0, 0, 0))
        game.screen.blit(text, (10, 10))
        
        text_output = font.render(f"Output: {output:.2f}", True, (255, 0, 0))
        game.screen.blit(text_output, (10, 50))
        
        speed_text = font.render(f"Speed: {speed_multiplier}x", True, (0, 100, 0))
        game.screen.blit(speed_text, (10, 90))
        
        pygame.display.flip()

        if done:
            pygame.time.wait(1500)
            obs = game.reset()
