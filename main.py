"""
Main Menu Interface
Provides user interface for training, testing, and managing AI models
"""

import pygame
import sys
import os
import platform
from game import FlappyBirdGame, SETTINGS
import train
from visualize import visualize_generation, play_best_ai


def clear_screen():
    """Clear console for clean UI"""
    os.system('cls' if platform.system() == 'Windows' else 'clear')


def customize_settings():
    """Menu to customize genetic algorithm parameters"""
    clear_screen()
    print("╔" + "═"*76 + "╗")
    print("║" + " "*19 + "⚙️  CUSTOMIZE EVOLUTION PARAMETERS" + " "*25 + "║")
    print("╚" + "═"*76 + "╝")
    
    # Parameter descriptions
    params = {
        "POP_SIZE": "Population size (networks/generation)",
        "HIDDEN_SIZE": "Hidden layer size",
        "GENERATIONS": "Number of generations",
        "MAX_STEPS": "Max steps per game",
        "MUTATION_STD": "Mutation intensity",
        "MUTATION_RATE": "Mutation rate (%)"
    }
    
    for key, desc in params.items():
        current = SETTINGS[key]
        new_value = input(f"\n{desc:.<45} [{current}]: ").strip()
        if new_value:
            try:
                SETTINGS[key] = float(new_value) if "." in new_value else int(new_value)
                print(f"✅ {key} = {SETTINGS[key]}")
            except:
                print(f"❌ Invalid value, kept {current}")
    
    input("\n✅ Parameters updated. [Press Enter]")


def show_menu():
    """Display the main menu"""
    clear_screen()
    print("╔" + "═"*76 + "╗")
    print("║" + " "*24 + "🎮 FLAPPYBIRD AI MENU" + " "*31 + "║")
    print("╠" + "═"*76 + "╣")
    print("║  1️  Play manually                                                          ║")
    print("║  2️  Train AI population (Genetic Algorithm)                                ║")
    print("║  3  Watch best AI play                                                     ║")
    print("║  4  Visualize full generation                                              ║")
    print("║  5  Customize evolution parameters                                         ║")
    print("║                                                                            ║")
    print("║  6  Save best AI model                                                     ║")
    print("║  7  Load saved model                                                       ║")
    print("║  8  Delete model                                                           ║")
    print("║                                                                            ║")
    print("║  0  Exit                                                                   ║")
    print("╚" + "═"*76 + "╝")


def Menu():
    """Main menu loop"""
    show_menu()
    choice = input("\n👉 Your choice: ").strip()

    if choice == '1':
        # Manual play mode
        print("\n🎮 Launching game... (SPACE to flap, ESC to quit)")
        input("   [Press Enter to start]")
        game = FlappyBirdGame()
        game.run()
        input("\n✅ Game finished. [Press Enter]")

    elif choice == '2':
        # Training mode
        print("\n⏱️  Calculating estimated time...")
        
        # Time estimation
        steps_per_eval = SETTINGS["MAX_STEPS"]
        total_evals = SETTINGS["GENERATIONS"] * SETTINGS["POP_SIZE"]
        total_steps = total_evals * steps_per_eval
        approx_time_seconds = total_steps * 0.00001  # Adjust based on your machine
        approx_time_minutes = approx_time_seconds / 60
        
        # Show configuration
        print(f"\n📊 Training Configuration:")
        print(f"   • Generations: {SETTINGS['GENERATIONS']}")
        print(f"   • Population per generation: {SETTINGS['POP_SIZE']}")
        print(f"   • Max steps per game: {SETTINGS['MAX_STEPS']}")
        print(f"   • Estimated time: ~{approx_time_minutes:.1f} min ({approx_time_seconds:.0f} sec)")
        print(f"\n💡 Status symbols during training:")
        print(f"   📈 = Improvement detected")
        print(f"   ➡️  = No change (stagnation)")
        print(f"   ⭐ = Model hit MAX_STEPS (excellent!)")
        
        # Confirmation
        confirm = input("\n▶️  Start training? (y/n): ").strip().lower()
        if confirm == 'y':
            print()
            train.evolutionary_training()
            if train.best_model_ever is not None:
                print(f"\n✅ Training successful!")
                print(f"   📈 Best score: {train.best_score_ever:.0f} pipes")
                input("   [Press Enter to continue]")
            else:
                print("⚠️ No best model found.")
                input("   [Press Enter to continue]")
        else:
            print("❌ Training cancelled.")
            input("   [Press Enter]")

    elif choice == '3':
        # Watch best AI play
        if train.best_model_ever is None:
            print("\n❌ No AI model available!")
            print("   → Run training first (option 2)")
            input("   [Press Enter]")
        else:
            print(f"\n🤖 Playing with best model ({train.best_score_ever:.0f} pipes)")
            print("   Controls: ← → ↑ ↓ adjust speed | ESC to quit")
            input("   [Press Enter to start]")
            play_best_ai()
            input("\n✅ Playback finished. [Press Enter]")

    elif choice == '4':
        # Visualize generation
        if not os.path.exists(os.path.join(os.path.dirname(__file__), "models", "generations")):
            print("\n❌ No training generations found!")
            print("   → Run training first (option 2)")
            input("   [Press Enter]")
        else:
            print("\n📺 Select a generation to visualize...")
            visualize_generation()
            input("\n✅ Visualization finished. [Press Enter]")

    elif choice == '5':
        # Customize settings
        customize_settings()

    elif choice == '6':
        # Save best model
        if train.best_model_ever is not None:
            name = input("\n💾 Model name (without extension): ").strip()
            if name:
                generation = input("   Generation (optional): ").strip() or None
                score = input("   Score (optional): ").strip() or None
                try:
                    if generation: 
                        generation = int(generation)
                    if score: 
                        score = float(score)
                    train.save_model_with_metadata(train.best_model_ever, name, generation, score)
                    input("   [Press Enter]")
                except:
                    print("❌ Invalid value")
                    input("   [Press Enter]")
            else:
                print("❌ Empty name, cancelled.")
        else:
            print("\n❌ No model to save!")
            input("   [Press Enter]")

    elif choice == '7':
        # Load model
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(models_dir, exist_ok=True)
        files = [f for f in os.listdir(models_dir) if f.endswith(".pth")]
        
        if not files:
            print("\n❌ No saved models found.")
            input("   [Press Enter]")
        else:
            print(f"\n📂 Available models ({len(files)}):")
            for i, file in enumerate(files, 1):
                print(f"   {i}. {file[:-4]}")  # Remove .pth extension
            try:
                idx = int(input("\n👉 Model number to load: ")) - 1
                if 0 <= idx < len(files):
                    name = files[idx].replace(".pth", "")
                    train.best_model_ever = train.load_model_with_metadata(name)
                    print(f"✅ Model loaded.")
                    input("   [Press Enter]")
                else:
                    print("❌ Invalid number.")
                    input("   [Press Enter]")
            except:
                print("❌ Invalid input.")
                input("   [Press Enter]")

    elif choice == '8':
        # Delete model
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(models_dir, exist_ok=True)
        files = [f for f in os.listdir(models_dir) if f.endswith(".pth")]
        
        if not files:
            print("\n❌ No models to delete.")
            input("   [Press Enter]")
        else:
            print(f"\n📂 Available models ({len(files)}):")
            for i, file in enumerate(files, 1):
                print(f"   {i}. {file[:-4]}")
            try:
                idx = int(input("\n👉 Model number to delete: ")) - 1
                if 0 <= idx < len(files):
                    confirm = input(f"⚠️  Confirm deletion? (y/n): ").strip().lower()
                    if confirm == 'y':
                        os.remove(os.path.join(models_dir, files[idx]))
                        metadata = os.path.join(models_dir, files[idx].replace(".pth", "_metadata.json"))
                        if os.path.exists(metadata):
                            os.remove(metadata)
                        print(f"✅ Model deleted.")
                        input("   [Press Enter]")
                    else:
                        print("❌ Deletion cancelled.")
                        input("   [Press Enter]")
                else:
                    print("❌ Invalid number.")
                    input("   [Press Enter]")
            except:
                print("❌ Invalid input.")
                input("   [Press Enter]")

    elif choice == '0':
        # Exit
        print("\n👋 Thank you for playing!")
        pygame.quit()
        sys.exit()
    
    else:
        print("\n❌ Invalid choice.")
        input("   [Press Enter]")

    # Recursive call to show menu again
    Menu()


if __name__ == "__main__":
    Menu()
