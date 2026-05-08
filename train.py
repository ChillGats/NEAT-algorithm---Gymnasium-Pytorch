"""
Genetic Algorithm Training Module
Implements the evolutionary training loop for the FlappyBird AI
"""

import random
import numpy as np
import torch
import os
import json
from game import FlappyBirdGame, SETTINGS
from model import NeuralNet, crossover, mutate
import pygame

best_model = None
best_model_ever = None
best_score_ever = float('-inf')

# Directory structure for saving trained models
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
GENERATIONS_FOLDER = os.path.join(MODELS_DIR, "generations")
os.makedirs(GENERATIONS_FOLDER, exist_ok=True)


def evolutionary_training():
    """
    Main genetic algorithm training loop.
    
    Process:
    1. Initialize population with random neural networks
    2. For each generation:
       - Evaluate all models on the game
       - Select elite performers
       - Create offspring via crossover and mutation
    3. Return best model found
    """
    global best_model_ever, best_score_ever
    
    # Initialize population with random networks
    population = [NeuralNet(hidden_size=SETTINGS["HIDDEN_SIZE"]) 
                  for _ in range(SETTINGS["POP_SIZE"])]
    
    ELITE_SIZE = 5  # Top 20% of population preserved each generation

    # Track statistics across all generations
    stats = {
        "best_scores": [],
        "avg_scores": [],
        "worst_scores": [],
        "improvements": [],
        "max_steps_hits": 0,
        "best_hit_max_steps_consecutive": 0,  # Track consecutive MAX_STEPS hits on best model
    }
    
    prev_best_score = 0

    # Training header
    print("\n" + "="*80)
    print(f"🚀 Starting Training ({SETTINGS['GENERATIONS']} generations)")
    print(f"   Population size: {SETTINGS['POP_SIZE']} | Elite: {ELITE_SIZE} | Max steps: {SETTINGS['MAX_STEPS']}")
    print("="*80 + "\n")

    # Main training loop - iterate through generations
    for gen in range(SETTINGS["GENERATIONS"]):
        # Adaptive mutation schedule: high exploration early, fine-tuning late
        # Cubic decay gives us more exploration in early generations
        progress = gen / SETTINGS["GENERATIONS"]
        decay_factor = 1 - (progress ** 3)  # Stays near 1.0 early, drops fast near end
        
        mutation_rate = max(0.05, SETTINGS["MUTATION_RATE"] * decay_factor)
        mutation_std = max(0.01, SETTINGS["MUTATION_STD"] * decay_factor)
        
        pygame.init()
        current_seed = random.randint(0, 999999)

        scores = []
        hit_max_steps_count = 0
        best_model_hit_max_steps = False
        
        # Evaluate each network in the population (fitness evaluation)
        for i, model in enumerate(population):
            # Use same seed for all models this generation
            random.seed(current_seed)
            np.random.seed(current_seed)
            torch.manual_seed(current_seed)
            
            # Play one game per model
            game = FlappyBirdGame(render_mode=None)
            obs = game.reset()
            steps = 0
            
            # Game loop until collision or max steps
            while not game.done and steps < SETTINGS["MAX_STEPS"]:
                # Convert observation to tensor
                x = torch.tensor(obs, dtype=torch.float32)
                # Get network output (0-1)
                output = model(x).item()
                # Threshold at 0.5: 1=flap, 0=no flap
                action = 1 if output > 0.5 else 0
                obs, reward, done, _ = game.step(action)
                steps += 1
            
            # Record fitness as pipes passed
            scores.append(game.score)
            
            # Track if model hit the time limit
            if steps >= SETTINGS["MAX_STEPS"] - 1:
                hit_max_steps_count += 1
                # Check if this is the best model of this generation
                if game.score == max(scores):
                    best_model_hit_max_steps = True

        # Sort population by fitness (best first)
        sorted_pairs = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
        
        # Extract elite models for breeding
        elites = [p[1] for p in sorted_pairs[:ELITE_SIZE]]
        
        best_score = sorted_pairs[0][0]
        worst_score = sorted_pairs[-1][0]
        avg_score = sum(scores) / len(scores)
        
        # Record statistics
        stats["best_scores"].append(best_score)
        stats["avg_scores"].append(avg_score)
        stats["worst_scores"].append(worst_score)
        stats["max_steps_hits"] += hit_max_steps_count

        # Determine status symbol for this generation
        if best_model_hit_max_steps:
            status = "⭐"  # Best model hit MAX_STEPS
            stats["best_hit_max_steps_consecutive"] += 1
        elif best_score > prev_best_score:
            status = "📈"  # Improvement detected
            stats["improvements"].append(gen)
            stats["best_hit_max_steps_consecutive"] = 0  # Reset counter
        else:
            status = "➡️"   # No improvement (stagnation)
            stats["best_hit_max_steps_consecutive"] = 0  # Reset counter

        # Print generation stats
        print(f"Gen {gen + 1:02d}/{SETTINGS['GENERATIONS']} | Avg: {avg_score:6.2f} | Top: {best_score:6.0f} | Worst: {worst_score:6.0f} | {status}")

        # Track all-time best model
        if best_score > best_score_ever:
            best_score_ever = best_score
            best_model_ever = elites[0]

        # Save checkpoint of current generation
        save_generation(population, gen + 1, current_seed)

        # Check for early stopping: best model hitting MAX_STEPS consistently
        if stats["best_hit_max_steps_consecutive"] >= 5:  # 5 generations of MAX_STEPS
            print("\n" + "-"*80)
            print(f"⚠️  Best model hit MAX_STEPS {stats['best_hit_max_steps_consecutive']} times consecutively!")
            print(f"    This means it can't improve without increasing MAX_STEPS.")
            print(f"    Current best score: {best_score_ever:.0f} pipes")
            print(f"    Current MAX_STEPS: {SETTINGS['MAX_STEPS']}")
            print("-"*80)
            choice = input("\n⏸️  Continue training? (y/n): ").strip().lower()
            if choice != 'y':
                print("✅ Training stopped by user.\n")
                break
            else:
                print("Continuing training...\n")
                stats["best_hit_max_steps_consecutive"] = 0  # Reset counter

        # Create next generation via breeding
        new_population = list(elites)  # Start with elites (no regression)
        
        # Fill rest of population with offspring
        while len(new_population) < SETTINGS["POP_SIZE"]:
            # Select two random parents from elite
            p1, p2 = random.sample(elites, 2)
            
            # Crossover: blend their weights
            child = crossover(p1, p2)
            
            # Mutation: add random noise
            child = mutate(child, mutation_rate=mutation_rate, std=mutation_std)
            
            new_population.append(child)

        population = new_population
        prev_best_score = best_score

    # Reset random seed after training
    random.seed() 
    
    # Print detailed final statistics
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED!")
    print("="*80)
    
    if stats["best_scores"]:
        print(f"\n📊 FINAL STATISTICS:")
        print(f"   📈 Best score achieved: {max(stats['best_scores']):.0f} pipes")
        print(f"   📉 Final generation average: {stats['avg_scores'][-1]:.2f} pipes")
        print(f"   💯 Overall average: {sum(stats['avg_scores'])/len(stats['avg_scores']):.2f} pipes")
        print(f"   ⭐ MAX_STEPS hit: {stats['max_steps_hits']} times total")
        
        # Progression metrics
        if len(stats["best_scores"]) > 1:
            initial_best = stats["best_scores"][0]
            final_best = stats["best_scores"][-1]
            improvement = final_best - initial_best
            improvement_pct = (improvement / max(1, initial_best)) * 100
            print(f"   🚀 Progression: Gen1 ({initial_best:.0f}) → Gen{len(stats['best_scores'])} ({final_best:.0f}) = +{improvement:.0f} pipes ({improvement_pct:.1f}%)")
        
        # Convergence metrics
        if len(stats["improvements"]) > 0:
            first_improvement_gen = stats["improvements"][0] + 1
            print(f"   ⚡ First improvement: Generation {first_improvement_gen}")
            print(f"   📍 Generations with improvement: {len(stats['improvements'])} / {len(stats['best_scores'])}")
        
        # Population diversity
        overall_best = max(stats["best_scores"])
        overall_worst = min(stats["worst_scores"])
        print(f"   🎯 Population spread: Best={overall_best:.0f}, Worst={overall_worst:.0f}, Range={overall_best - overall_worst:.0f}")
    
    print("="*80 + "\n")
    
    return best_model_ever


def evaluate(_model):
    """
    Evaluate a single model's fitness by running one game.
    
    Args:
        _model: Neural network model to evaluate
        
    Returns:
        Fitness score (number of pipes passed)
    """
    game = FlappyBirdGame(render_mode=None)
    obs = game.reset()
    steps = 0
    
    # Play until collision or max steps
    while not game.done and steps < SETTINGS["MAX_STEPS"]:
        x = torch.tensor(obs, dtype=torch.float32)
        output = _model(x).item()
        action = 1 if output > 0.5 else 0
        obs, reward, done, _ = game.step(action)
        steps += 1
    
    return game.score


def save_generation(models, generation, seed):
    """
    Save all models from a generation as checkpoints.
    
    Args:
        models: List of neural networks
        generation: Generation number
        seed: Random seed used for reproducibility
    """
    gen_folder = os.path.join(GENERATIONS_FOLDER, f"gen_{generation:03d}")
    os.makedirs(gen_folder, exist_ok=True)
    
    # Save seed for reproducibility
    with open(os.path.join(gen_folder, "seed.json"), "w") as f:
        json.dump({"seed": seed}, f)
    
    # Save each model
    for i, _model in enumerate(models):
        filename = os.path.join(gen_folder, f"model_{i:03d}.pth")
        torch.save(_model.state_dict(), filename)


def load_generation_models(generation):
    """
    Load all models from a specific generation.
    
    Args:
        generation: Generation number to load
        
    Returns:
        Tuple of (models list, seed used)
    """
    gen_folder = os.path.join(GENERATIONS_FOLDER, f"gen_{generation:03d}")
    if not os.path.exists(gen_folder):
        print(f"❌ Generation {generation} not found.")
        return None, None
    
    # Load seed
    with open(os.path.join(gen_folder, "seed.json"), "r") as f:
        data = json.load(f)
        seed = data.get("seed", None)
    
    # Load all models from generation
    model_files = sorted([f for f in os.listdir(gen_folder) if f.endswith(".pth")])
    models = []
    for f in model_files:
        _model = NeuralNet()
        _model.load_state_dict(torch.load(os.path.join(gen_folder, f)))
        models.append(_model)
    
    return models, seed


def save_model_with_metadata(_model, name, generation=None, score=None):
    """
    Save a model with metadata (generation, score, settings).
    
    Args:
        _model: Neural network to save
        name: Model name (without extension)
        generation: Which generation it came from
        score: Best score achieved
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    filename = os.path.join(MODELS_DIR, f"{name}.pth")
    metadata = os.path.join(MODELS_DIR, f"{name}_metadata.json")
    
    # Save model weights
    torch.save(_model.state_dict(), filename)
    
    # Save metadata
    metadata_dict = {
        "name": name,
        "generation": generation,
        "score": score,
        "settings": SETTINGS,
    }
    with open(metadata, "w") as f:
        json.dump(metadata_dict, f, indent=4)
    
    print(f"✅ Model saved to {filename} with metadata.")


def load_model_with_metadata(name):
    """
    Load a model and display its metadata.
    
    Args:
        name: Model name (without extension)
        
    Returns:
        Loaded neural network model
    """
    filename = os.path.join(MODELS_DIR, f"{name}.pth")
    metadata = os.path.join(MODELS_DIR, f"{name}_metadata.json")
    
    if not os.path.exists(filename):
        print("❌ Model not found.")
        return None
    
    # Load model
    _model = NeuralNet()
    _model.load_state_dict(torch.load(filename))
    
    # Display metadata if available
    if os.path.exists(metadata):
        with open(metadata, "r") as f:
            data = json.load(f)
        print("📊 Model metadata:")
        for key, value in data.items():
            if key != "settings":
                print(f"   {key}: {value}")
    else:
        print("⚠️ No metadata found.")
    
    return _model
