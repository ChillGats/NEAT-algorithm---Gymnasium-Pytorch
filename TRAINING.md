# Training Guide - FlappyBird AI

This guide explains how the AI learns to play FlappyBird using genetic algorithms.

## Overview

A **Genetic Algorithm (GA)** is an optimization method inspired by natural selection. It evolves solutions over generations by:
- **Population**: Starting with random neural networks.
- **Fitness**: Evaluating performance (pipes passed).
- **Selection**: Keeping the best performers.
- **Reproduction**: Breeding new networks with mutations.

### Why Use Genetic Algorithms?
- No labeled data required.
- Diverse strategies emerge.
- Scalable and interpretable.

---

## Training Process

### Steps Per Generation:
1. **Evaluate**: Each network plays the game, and scores are recorded.
2. **Select**: Top 5 networks are preserved.
3. **Reproduce**: 20 offspring are created via crossover and mutation.
4. **Repeat**: Continue for multiple generations.

### Typical Progression:
- Early: Random exploration.
- Mid: Rapid improvement.
- Late: Fine-tuning.

---

## Key Parameters

- **Population Size**: 25 networks (balanced for diversity and speed).
- **Fitness**: Measured by pipes passed.
- **Mutation**: Adaptive rate for exploration and refinement.

---

## Tips for Success

- Monitor training progression to identify stagnation.
- Use early stopping if no improvement is observed.
- Adjust parameters like `POP_SIZE` and `MAX_STEPS` for faster or higher-quality results.

For detailed configuration, see `game.py` settings.
