import pygame
import sys
import os
from game import FlappyBirdGame, SETTINGS
import train
from visualize import visualize_generation, play_best_ai

def customize_settings():
    print("\n--- Personnalisation des paramètres d'évolution ---")
    for key in ["POP_SIZE", "HIDDEN_SIZE", "GENERATIONS", "MAX_STEPS", "MUTATION_STD", "MUTATION_RATE"]:
        current = SETTINGS[key]
        new_value = input(f"{key} (actuel: {current}) : ")
        if new_value:
            if "." in new_value:
                SETTINGS[key] = float(new_value)
            else:
                SETTINGS[key] = int(new_value)
    print("✅ Paramètres mis à jour.")

def Menu():
    print("\n=== MENU PRINCIPAL ===")
    print("1 - jouer une partie")
    print("2 - entrainer une population IA")
    print("3 - faire jouer la meilleure IA")
    print("4 - personnaliser les paramètres d'évolution")
    print("5 - visualiser une génération complète")
    print("6 - sauvegarder la meilleure IA")
    print("7 - charger la meilleure IA")
    print("10 - supprimer un modèle existant")
    print("0 - quitter le programme")

    choice = input("Choix : ")

    if choice == '1':
        game = FlappyBirdGame()
        game.run()
    elif choice == '2':
        # Estimate time: roughly 0.005 seconds per evaluation step (tune based on your machine)
        steps_per_eval = SETTINGS["MAX_STEPS"]
        total_evals = SETTINGS["GENERATIONS"] * SETTINGS["POP_SIZE"]
        total_steps = total_evals * steps_per_eval
        approx_time_seconds = total_steps * 0.00001  # Adjust multiplier based on testing
        approx_time_minutes = approx_time_seconds / 60
        confirm = input(f"L'entraînement peut prendre environ {approx_time_minutes:.1f} minutes ({approx_time_seconds:.0f} secondes). Continuer ? (o/n) ")
        if confirm.lower() == 'o':
            train.evolutionary_training()
            if train.best_model_ever is not None:
                print(f"✅ Meilleur modèle disponible avec score {train.best_score_ever}.")
            else:
                print("⚠️ Aucun meilleur modèle trouvé pendant l'entraînement.")
    elif choice == '3':
        play_best_ai()
    elif choice == '4':
        customize_settings()
    elif choice == '5':
        visualize_generation()
    elif choice == '6':
        global best_model_ever
        if train.best_model_ever is not None:
            name = input("Nom du modèle à sauvegarder (sans extension) : ")
            generation = input("Génération (optionnel) : ") or None
            score = input("Score (optionnel) : ") or None
            if generation: generation = int(generation)
            if score: score = float(score)
            train.save_model_with_metadata(train.best_model_ever, name, generation, score)
        else:
            print("❌ Aucun modèle à sauvegarder.")
    elif choice == '7':
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(models_dir, exist_ok=True)
        files = [f for f in os.listdir(models_dir) if f.endswith(".pth")]
        if not files:
            print("❌ Aucun modèle trouvé.")
        else:
            for i, file in enumerate(files):
                print(f"{i + 1} - {file}")
            idx = int(input("Numéro du modèle à charger : ")) - 1
            if 0 <= idx < len(files):
                name = files[idx].replace(".pth", "")
                train.best_model_ever = train.load_model_with_metadata(name)
                print(f"✅ Modèle {files[idx]} chargé.")
            else:
                print("❌ Numéro invalide.")
    elif choice == '10':
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(models_dir, exist_ok=True)
        files = [f for f in os.listdir(models_dir) if f.endswith(".pth")]
        if not files:
            print("❌ Aucun modèle à supprimer.")
        else:
            for i, file in enumerate(files):
                print(f"{i + 1} - {file}")
            idx = int(input("Numéro du modèle à supprimer : ")) - 1
            if 0 <= idx < len(files):
                os.remove(os.path.join(models_dir, files[idx]))
                metadata = os.path.join(models_dir, files[idx].replace(".pth", "_metadata.json"))
                if os.path.exists(metadata):
                    os.remove(metadata)
                print(f"✅ Modèle {files[idx]} supprimé.")
            else:
                print("❌ Numéro invalide.")
    elif choice == '0':
        print("Fermeture du programme.")
        pygame.quit()
        sys.exit()
    else:
        print("Choix non reconnu.")

    Menu()

if __name__ == "__main__":
    Menu()
