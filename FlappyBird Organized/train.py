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

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
GENERATIONS_FOLDER = os.path.join(MODELS_DIR, "generations")
os.makedirs(GENERATIONS_FOLDER, exist_ok=True)

def evolutionary_training():
    global best_model_ever, best_score_ever
    population = [NeuralNet(hidden_size=SETTINGS["HIDDEN_SIZE"]) for _ in range(SETTINGS["POP_SIZE"])]
    
    # On garde les 5 meilleurs à chaque fois (20% de la population de 25)
    ELITE_SIZE = 5 

    for gen in range(SETTINGS["GENERATIONS"]):
        pygame.init()
        current_seed = random.randint(0, 999999)

        scores = []
        for models in population:
            random.seed(current_seed)
            np.random.seed(current_seed)
            torch.manual_seed(current_seed)
            scores.append(evaluate(models))

        # --- NOUVELLE SÉLECTION (LE TOURNOI) ---
        # On trie la population du meilleur au pire score
        sorted_pairs = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
        
        # On extrait les X meilleurs (les Élites)
        elites = [p[1] for p in sorted_pairs[:ELITE_SIZE]]
        
        best_score = sorted_pairs[0][0]
        avg_score = sum(scores) / len(scores)

        print(f"Génération {gen + 1:02d}/{SETTINGS['GENERATIONS']} | Moy: {avg_score:6.2f} | Top: {best_score:6.2f}")

        # Sauvegarde du champion absolu
        if best_score > best_score_ever:
            best_score_ever = best_score
            best_model_ever = elites[0]

        save_generation(population, gen + 1, current_seed)

        # --- REPRODUCTION ---
        # La nouvelle génération commence avec nos élites intactes (pour ne jamais régresser)
        new_population = list(elites)

        # On remplit le reste de la population avec des enfants
        while len(new_population) < SETTINGS["POP_SIZE"]:
            # On choisit 2 parents au hasard parmi l'élite
            p1, p2 = random.sample(elites, 2)
            
            # Croisement
            child = crossover(p1, p2)
            
            # Mutation (Rate = 20% des gènes)
            child = mutate(child, mutation_rate=SETTINGS["MUTATION_RATE"], std=SETTINGS["MUTATION_STD"])
            
            new_population.append(child)

        population = new_population

    random.seed() 
    print("\n✅ Entraînement terminé !")
    return best_model_ever

def evaluate(_model):
    """Fitness: number of pipes passed (game.score)"""
    game = FlappyBirdGame(render_mode=None)
    obs = game.reset()
    steps = 0
    while not game.done and steps < SETTINGS["MAX_STEPS"]:
        x = torch.tensor(obs, dtype=torch.float32)
        output = _model(x).item()
        action = 1 if output > 0.5 else 0
        obs, reward, done, _ = game.step(action)
        steps += 1
    return game.score  # Real fitness: pipes passed

def save_generation(models, generation, seed):
    gen_folder = os.path.join(GENERATIONS_FOLDER, f"gen_{generation:03d}")
    os.makedirs(gen_folder, exist_ok=True)
    with open(os.path.join(gen_folder, "seed.json"), "w") as f:
        json.dump({"seed": seed}, f)
    for i, _model in enumerate(models):
        filename = os.path.join(gen_folder, f"model_{i:03d}.pth")
        torch.save(_model.state_dict(), filename)

def load_generation_models(generation):
    gen_folder = os.path.join(GENERATIONS_FOLDER, f"gen_{generation:03d}")
    if not os.path.exists(gen_folder):
        print(f"❌ Génération {generation} non trouvée.")
        return None, None
    with open(os.path.join(gen_folder, "seed.json"), "r") as f:
        data = json.load(f)
        seed = data.get("seed", None)
    model_files = sorted([f for f in os.listdir(gen_folder) if f.endswith(".pth")])
    models = []
    for f in model_files:
        _model = NeuralNet()
        _model.load_state_dict(torch.load(os.path.join(gen_folder, f)))
        models.append(_model)
    return models, seed

def save_model_with_metadata(_model, name, generation=None, score=None):
    """Save model with metadata"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    filename = os.path.join(MODELS_DIR, f"{name}.pth")
    metadata = os.path.join(MODELS_DIR, f"{name}_metadata.json")
    torch.save(_model.state_dict(), filename)
    
    metadata_dict = {
        "name": name,
        "generation": generation,
        "score": score,
        "settings": SETTINGS,
        "timestamp": str(torch.tensor(0).device)  # Placeholder for timestamp
    }
    with open(metadata, "w") as f:
        json.dump(metadata_dict, f, indent=4)
    print(f"✅ Modèle sauvegardé sous {filename} avec métadonnées.")

def load_model_with_metadata(name):
    """Load model and display metadata"""
    filename = os.path.join(MODELS_DIR, f"{name}.pth")
    metadata = os.path.join(MODELS_DIR, f"{name}_metadata.json")
    if not os.path.exists(filename):
        print("❌ Modèle non trouvé.")
        return None
    
    _model = NeuralNet()
    _model.load_state_dict(torch.load(filename))
    
    if os.path.exists(metadata):
        with open(metadata, "r") as f:
            data = json.load(f)
        print("📊 Métadonnées du modèle :")
        for key, value in data.items():
            if key == "settings":
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")
    else:
        print("⚠️ Aucune métadonnée trouvée.")
    
    return _model
