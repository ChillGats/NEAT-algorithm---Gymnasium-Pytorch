# FlappyBird AI - Genetic Algorithm

An AI trained with genetic algorithms to play FlappyBird autonomously.

## Features
- **Genetic Algorithm**: Evolves neural networks over generations.
- **Neural Network**: Compact architecture with 6D input and 1 output.
- **Real-time Visualization**: Watch training progress.
- **Configurable**: Adjust parameters like population size and mutation rate.
- **Model Persistence**: Save and load trained models.

---

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run the Game
```bash
python main.py
```

### Options
1. **Play manually**: Control the bird yourself.
2. **Train AI**: Run genetic algorithm training.
3. **Watch AI**: See the trained model in action.
4. **Visualize Training**: Observe all networks in a generation.

---

## How It Works

1. **Initialize**: Start with random networks.
2. **Evaluate**: Measure fitness by pipes passed.
3. **Evolve**: Select top performers, breed, and mutate.
4. **Repeat**: Improve over generations.

---

## Project Structure
```
FlappyBird/
├── main.py          # Entry point
├── game.py          # Game engine
├── model.py         # Neural network & GA logic
├── train.py         # Training loop
├── visualize.py     # Visualization tools
├── models/          # Saved models
├── README.md        # This file
├── TRAINING.md      # Training guide
├── requirements.txt # Dependencies
```

For detailed training instructions, see `TRAINING.md`.
