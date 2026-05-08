# FlappyBird Genetic Algorithm

A neural network AI trained using genetic algorithms to play FlappyBird autonomously. Starting from scratch, the AI learns to navigate pipes with increasing efficiency over generations.

## 📊 Features

- **Genetic Algorithm Training**: 50+ generations with adaptive mutation schedules
- **Neural Network**: 6-layer input → 32 hidden → 1 output binary classifier
- **Real-time Visualization**: Watch all birds train simultaneously at adjustable speeds (1x-960x)
- **Persistent Models**: Save/load trained models with generation tracking
- **Speed Control**: Play or visualize at any speed (← → ↑ ↓ keys)
- **Metdata Tracking**: Automatic logging of best scores and timestamps

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Train a Model
```bash
python main.py
# Select option 1: Train
```

### Visualize Training
```bash
python main.py
# Select option 2: Visualize Generation
```

### Play Best Model
```bash
python main.py
# Select option 3: Play Best AI
```

### Manual Play
```bash
python main.py
# Select option 10: Play the Game
```

## 🎮 Controls

### During Training Visualization
- **← Arrow**: Halve speed
- **→ Arrow**: Double speed  
- **↑ Arrow**: +1x speed
- **↓ Arrow**: -1x speed
- **ESC**: Exit

### Bird Controls (Manual Play)
- **SPACE**: Flap
- **ESC**: Exit

## 🧠 How It Works

### Observation Space (6D)
Each bird observes:
- `bird_y_normalized`: Vertical position (0-1)
- `velocity_normalized`: Fall speed (-1 to 1)
- `next_pipe_x_distance`: Distance to next pipe (0-1)
- `gap_center_y`: Center of gap between pipes (0-1)
- `gap_top_y`: Top of gap (0-1)
- `gap_bottom_y`: Bottom of gap (0-1)

### Network Architecture
```
Input (6) 
  ↓ [6×32] 
Tanh Activation
  ↓ [32×1]
Sigmoid Output (0=no flap, 1=flap)
```

### Genetic Algorithm
1. **Population**: 25 neural networks per generation
2. **Evaluation**: Each model plays one game, fitness = pipes passed + (time_steps × 0.1)
3. **Selection**: Top 5 models (elite) preserved
4. **Breeding**: Elite models bred with crossover (50/50 weight blending)
5. **Mutation**: Adaptive Gaussian noise (starts high, decreases over generations)
6. **Repetition**: 50 generations total

### Fitness Function
```python
fitness = game.score × 10 + steps_survived × 0.1
```
Prioritizes distance traveled, rewards survivorship.

## 📈 Expected Results
( depends on the max steps )
After 50 generations:
- **Gen 1**: Average ~0 pipes, best ~0-1
- **Gen 10**: Average ~0.2 pipes, best ~1
- **Gen 30**: Average ~25 pipes, best ~50+
- **Gen 50**: Average ~150+ pipes, best ~inf

Best model typically reaches **inf pipes** without collision.

## 📂 Project Structure

```
FlappyBird/
├── main.py              Main menu interface
├── game.py              Game engine & physics
├── model.py             Neural network & GA operators
├── train.py             Training loop
├── visualize.py         Visualization modes
└── models/
    └── generations/     Trained model checkpoints
        ├── gen_001/
        ├── gen_002/
        └── ... (auto-generated)
```

## 🔧 Advanced Configuration

Edit `game.py` SETTINGS dict to customize:
```python
SETTINGS = {
    "SCREEN_WIDTH": 800,
    "SCREEN_HEIGHT": 600,
    "FPS": 60,
    "PIPE_GAP": 150,          # Space between pipes
    "PIPE_SPEED": 5,          # Horizontal scroll speed
    "PIPE_FREQUENCY": 90,     # Frames between pipe spawns
    "GRAVITY": 0.5,
    "FLAP_POWER": 12,
    
    # GA Settings
    "HIDDEN_SIZE": 32,
    "POP_SIZE": 25,
    "GENERATIONS": 50,
    "MUTATION_RATE": 0.2,
    "MUTATION_STD": 0.05,
}
```

## 📝 Algorithm Details

### Adaptive Mutation Schedule
Mutation decreases over generations (exploration → exploitation):
```python
mutation_rate = base_rate × (1 - gen/max_gen)^0.5
mutation_std = base_std × (1 - gen/max_gen)
```

### Elite Preservation
Top 5 models are always preserved and re-evaluated with same seed for consistency.

### Crossover Mechanism
New models blend parent weights 50/50:
```python
child_weight = random_mask * parent1_weight + (1 - random_mask) * parent2_weight
```

## 🐛 Troubleshooting

**No models found error?**
- Run training first (option 1) to generate models
- Models saved to `models/generations/gen_XXX/`

**Speed not changing?**
- Hold arrow key (not just tap)
- Speed displays as "Speed: Nx" in top-left

**Low performance on visualize?**
- Reduce speed multiplier (← key) for smoother rendering
- Close other applications

## 💡 Future Improvements

- [ ] Add curriculum learning (gradually increase difficulty)
- [ ] Implement different mutation operators (polynomial, uniform)
- [ ] Train on multiple random seeds for robustness
- [ ] Add score graph tracking over generations
- [ ] Implement Q-learning comparison baseline
- [ ] Convert to web app with Pygame-to-Canvas


## 🙏 Acknowledgments

Based on FlappyBird game mechanics with custom genetic algorithm implementation using PyTorch.
base project created by **ChillGats** (creator of the repo)

