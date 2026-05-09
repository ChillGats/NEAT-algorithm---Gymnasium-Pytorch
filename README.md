# 🎮 FlappyBird AI - Genetic Algorithm Evolution

A neural network AI trained using genetic algorithms to play FlappyBird autonomously. The AI learns to navigate endless pipes with increasing efficiency over generations, achieving **infinite scores** with proper training.

## ✨ Key Highlights

- **🧬 Genetic Algorithm**: Population-based evolutionary training with elite preservation
- **🧠 Neural Network**: Compact 6D input → 32 hidden → 1 output binary classifier
- **📈 Unlimited Scoring**: No hard pipe limit - AI can theoretically play forever
- **⏱️ Smart Early Stopping**: Automatically detects when AI can't improve due to time limits
- **🎯 Real-time Visualization**: Watch all 25 birds train simultaneously at adjustable speeds (1x-960x)
- **💾 Model Persistence**: Save/load trained models with full metadata
- **⚙️ Highly Configurable**: Tune population size, mutations, steps, and generations
- **📊 Detailed Statistics**: Track progression, convergence, population diversity

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run the Game
```bash
cd FlappyBird
python main.py
```

### Main Menu Options
1. **Play manually** - Control the bird yourself (SPACE to flap)
2. **Train AI** - Run genetic algorithm training (50 generations by default)
3. **Watch best AI play** - See the trained model in action (speed adjustable)
4. **Visualize generation** - Watch all 25 birds from a generation simultaneously
5. **Customize parameters** - Adjust population size, steps, mutations, etc.
6-8. **Save/Load/Delete models** - Manage your trained AI models

## 🧬 How It Works

### The Algorithm

```
Generation 1 (random) → Evaluation → Selection → Breeding → Mutation → Generation 2
                           ↓
                    Best: 5 pipes
                    Average: 2

                                                                      → Generation 50
                                                                        Best: ∞ pipes
                                                                        Average: 200+
```

### Key Components

**1. Neural Network** (Tiny but Effective)
- Input: 6D observation (bird position, velocity, pipe distance, gap position)
- Hidden: 32 neurons with Tanh activation
- Output: 1 sigmoid neuron (0=no flap, 1=flap)
- Total: ~224 parameters

**2. Population** (25 Networks)
- 5 elite networks preserved each generation (top 20%)
- 20 offspring created from elite via crossover + mutation
- Different network = different flying strategy

**3. Fitness Evaluation**
- Each network plays one game (up to MAX_STEPS)
- Fitness = pipes passed (higher = better)
- Score is measured in pipes successfully navigated

**4. Selection & Breeding**
- Top 5 performers become parents
- Crossover: Random 50/50 weight blending from two parents
- Mutation: Adaptive Gaussian noise (high early exploration → low fine-tuning)

### Adaptive Mutation Schedule
```python
Early generations:   High mutation (explore solution space)
Middle generations:  Medium mutation (refine strategies)
Late generations:    Low mutation (fine-tune best strategies)
```

Uses cubic decay: `mutation_rate = base × (1 - (gen/max_gen)³)`

## 📊 Training Progression

**Typical Training Timeline** (depends on MAX_STEPS & POP_SIZE):

| Generation | Best Score | Avg Score | Status |
|------------|-----------|-----------|--------|
| 1-5 | 1-2 pipes | 0-1 | Random exploration |
| 5-15 | 2 pipes | 2-5 | Learning begins |
| **10-25** | **50-150 pipes** | **10-15** | **Take-off phase (most important)** |
| 25-40 | 150-300 pipes | 80-130 | Refinement |
| 40-50 | 300+ pipes | 150-250 | Convergence (hitting MAX_STEPS) |

**Key Insight**: Most progression happens at **generations 10-25** regardless of other settings.

### Status Symbols During Training

- `📈` = Improvement detected (best score increased)
- `➡️` = No change (stagnation)
- `⭐` = Best model hit MAX_STEPS (time limit reached, not skill limit)

### Smart Early Stopping

When best model hits MAX_STEPS for **5 consecutive generations**:
```
⚠️  Best model hit MAX_STEPS 5 times consecutively!
    This means it can't improve without increasing MAX_STEPS.
    Current best score: 245 pipes
    Current MAX_STEPS: 15000

⏸️  Continue training? (y/n):
```

This prevents wasting time on training when the AI has reached the time limit.

## 🎮 Controls

### During Training Visualization
- `← Arrow`: Halve playback speed
- `→ Arrow`: Double playback speed
- `↑ Arrow`: +1x playback speed
- `↓ Arrow`: -1x playback speed
- `ESC`: Exit

### Manual Play
- `SPACE`: Flap (make bird go up)
- `ESC`: Exit

## 📁 Project Structure

```
FlappyBird/
├── main.py              # Menu interface & entry point
├── game.py              # Game engine, physics, pipe generation
├── model.py             # Neural network architecture & GA operators
├── train.py             # Genetic algorithm training loop
├── visualize.py         # Visualization modes (generation, best model)
├── models/              # Saved trained models
│   └── generations/     # Checkpoints (gen_001, gen_002, ...)
│       └── gen_001/     # Gen 1 models + seed.json
├── README.md            # This file
├── TRAINING.md          # Detailed training guide
├── requirements.txt     # Python dependencies
└── .gitignore          # Git ignore rules
```

## ⚙️ Configuration

Edit `game.py` SETTINGS to customize:

```python
SETTINGS = {
    # Screen & Game Physics
    "SCREEN_WIDTH": 1000,
    "SCREEN_HEIGHT": 600,
    "PIPE_GAP": 175,           # Space between pipes (larger = easier)
    "PIPE_VELOCITY": 5,        # Pipe scroll speed (higher = harder)
    "GRAVITY": 0.5,
    "FLAP_STRENGTH": -10,
    "BIRD_SIZE": 30,
    "FPS": 60,
    
    # Genetic Algorithm
    "MAX_STEPS": 15000,        # Max steps per game (~1 pipe every 50 steps)
    "GENERATIONS": 50,         # Number of GA generations
    "POP_SIZE": 25,            # Networks per generation
    "HIDDEN_SIZE": 32,         # Hidden layer neurons
    "MUTATION_RATE": 0.2,      # 20% of weights mutated per offspring
    "MUTATION_STD": 0.05,      # Gaussian noise standard deviation
}
```

### Recommended Setups

**⚡ Fast Training** (2-3 min per run)
```python
MAX_STEPS: 5000
GENERATIONS: 50
POP_SIZE: 25
# Result: Best ~100 pipes, avg ~30
```

**⚙️ Balanced** (7-10 min per run)
```python
MAX_STEPS: 15000
GENERATIONS: 50
POP_SIZE: 25
# Result: Best ~300 pipes, avg ~80-100
```

**🎯 High Quality** (20-25 min per run)
```python
MAX_STEPS: 50000
GENERATIONS: 50
POP_SIZE: 25
# Result: Best 500-1000+ pipes, avg ~200+
```

**🚀 More Diversity** (more generations, same time)
```python
MAX_STEPS: 5000
GENERATIONS: 100
POP_SIZE: 25
# Result: More generations to find solutions, handles local optima better
```

## 📈 Performance Tuning

**To get higher scores:**
- ↑ Increase `MAX_STEPS` (gives more time to learn, but slower training)
- ↑ Increase `GENERATIONS` (more evolution time)
- ↑ Increase `POP_SIZE` (more genetic diversity)

**For faster training:**
- ↓ Decrease `MAX_STEPS` (less time per evaluation)
- ↓ Decrease `GENERATIONS` (fewer iterations)
- ↓ Decrease `POP_SIZE` (fewer networks per generation)

**For better quality:**
- ↓ Decrease `MUTATION_RATE` (more stable breeding)
- ↓ Decrease `MUTATION_STD` (smaller random changes)
- ↑ Increase population but same steps (try `POP_SIZE: 50`)

## 📊 Example Training Output

```
🚀 Starting Training (50 generations)
   Population size: 25 | Elite: 5 | Max steps: 15000

Gen 01/50 | Avg:   2.45 | Top:      6 | Worst:      0 | ➡️
Gen 02/50 | Avg:   3.12 | Top:      8 | Worst:      1 | 📈
Gen 03/50 | Avg:   4.56 | Top:     12 | Worst:      2 | 📈
...
Gen 15/50 | Avg:  45.23 | Top:    127 | Worst:     12 | ⭐
Gen 16/50 | Avg:  48.10 | Top:    135 | Worst:     14 | 📈
...
Gen 50/50 | Avg: 182.45 | Top:    315 | Worst:     95 | ⭐

================================================================================
✅ TRAINING COMPLETED!
================================================================================

📊 FINAL STATISTICS:
   📈 Best score achieved: 315 pipes
   📉 Final generation average: 182.45 pipes
   💯 Overall average: 98.23 pipes
   ⭐ MAX_STEPS hit: 28 times total
   🚀 Progression: Gen1 (6) → Gen50 (315) = +309 pipes (5150.0%)
   ⚡ First improvement: Generation 2
   📍 Generations with improvement: 32 / 50
   🎯 Population spread: Best=315, Worst=5, Range=310
```

## 🧠 Neural Network Details

### Observation Space (6D Input)
```python
obs = [
    bird_y_normalized,              # 0-1: bird vertical position
    velocity_normalized,            # -1-1: falling speed
    pipe_distance_normalized,       # 0-1: distance to next pipe
    gap_center_normalized,          # 0-1: gap center position relative to bird
    gap_top_normalized,            # 0-1: gap top edge position
    gap_bottom_normalized          # 0-1: gap bottom edge position
]
```

### Network Architecture
```
Input (6 values)
    ↓
Dense Layer [6×32] + Bias
    ↓
Tanh Activation (introduces non-linearity)
    ↓
Dense Layer [32×1] + Bias
    ↓
Sigmoid Activation (output between 0-1)
    ↓
Decision: if output > 0.5: flap, else: do nothing
```

Total parameters: 6×32 + 32 + 32×1 + 1 = **≈224 weights**

### Why This Architecture?
- **Compact**: Very few parameters (fast training)
- **Effective**: 6D observations sufficient for FlappyBird
- **Non-linear**: Tanh hidden layer captures complex pipe-dodging patterns
- **Binary output**: Perfect for 2-action problem (flap/no-flap)

## 🔬 Genetic Operators

### Crossover (50/50 Blending)
```
Parent 1 weights: [0.5, -0.3, 0.8, 0.2, ...]
Parent 2 weights: [0.1,  0.6, -0.2, 0.9, ...]
Random mask:      [1,    0,   1,    0,   ...]

Child = mask × Parent1 + (1 - mask) × Parent2
      = [0.5,  0.6,   0.8,   0.9,   ...]
```
Inherits strengths from both parents.

### Mutation (Adaptive Gaussian Noise)
```
For each weight:
    if random() < mutation_rate:
        weight += Gaussian_noise(μ=0, σ=mutation_std)
    else:
        weight unchanged
```

Mutation rate and std decrease over generations (exploration → exploitation).

## 🐛 Troubleshooting

**Q: "No AI model available!"**
- A: Run training first (option 2 in menu)

**Q: "AI keeps hitting MAX_STEPS"**
- A: It's actually playing well! Increase MAX_STEPS to see if it improves further

**Q: "Training is too slow"**
- A: Reduce MAX_STEPS (5000) or POP_SIZE (15)

**Q: "AI doesn't improve much"**
- A: Wait for generation 10-25 (take-off phase), or increase GENERATIONS

**Q: "Best score seems stuck"**
- A: That's normal! Once AI hits MAX_STEPS consistently, it can't improve without more time

**Q: "How do I know training is working?"**
- A: Watch for `📈` symbols during training. Should see progress from gen 1-25

## 📚 Further Learning

- See `TRAINING.md` for detailed algorithm explanation
- Check `game.py` for physics implementation
- Review `model.py` for neural network code
- Examine `train.py` for GA loop specifics

## 🎓 Key Insights

1. **Take-off Phase (Gen 10-25)**: Most learning happens here regardless of settings
2. **MAX_STEPS Matters**: Determines max achievable score (1 pipe ≈ 50 steps)
3. **Population Diversity**: Larger POP_SIZE helps find better solutions
4. **Mutation Schedule**: Cubic decay provides good balance of exploration/exploitation
5. **Elite Preservation**: Prevents regression, ensures monotonic improvement
6. **Early Stopping**: Check for 5 consecutive ⭐ symbols to stop wasting time

## 📝 License

MIT License - Feel free to use, modify, and share!

## 🙏 Acknowledgments

Built with PyTorch, PyGame, and genetic algorithm principles. Inspired by classic FlappyBird games and evolutionary computation research.

---

**Ready to train an AI?** Run `python main.py` and select option 2! 🚀
