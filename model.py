import torch
import torch.nn as nn
import random
from game import SETTINGS

# Device detection: Use GPU if available, otherwise CPU
#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu') # ! IDK why but GPU is slower than CPU for this project
if torch.cuda.is_available():
    print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)} | Computing on CUDA")
else : 
    print("⚠️  No GPU detected. Computing on CPU (this will be slower).")

class NeuralNet(nn.Module):
    def __init__(self, input_size=6, hidden_size=None):
        super().__init__()
        hidden_size = hidden_size if hidden_size else SETTINGS["HIDDEN_SIZE"]
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        # Move network to device (GPU or CPU)
        self.net.to(DEVICE)

    def forward(self, x):
        # Ensure input is on the correct device
        x = x.to(DEVICE)
        return self.net(x)

# === NOUVEAUX OPÉRATEURS GÉNÉTIQUES ===

def crossover(parent1, parent2):
    """Blend weights from two neural networks (50/50 random mask)."""
    child = NeuralNet(hidden_size=SETTINGS["HIDDEN_SIZE"])
    state1 = parent1.state_dict()
    state2 = parent2.state_dict()
    child_state = {}
    
    with torch.no_grad():
        for key in state1:
            # Create random 0/1 mask to select which parent's weights to use
            mask = (torch.rand_like(state1[key]) > 0.5).to(DEVICE)
            # Blend genes from both parents
            child_state[key] = torch.where(mask, state1[key], state2[key])
            
    child.load_state_dict(child_state)
    return child

def mutate(model, mutation_rate=0.2, std=None):
    """Mutate only mutation_rate% of weights to avoid destroying learning."""
    std = std if std else SETTINGS["MUTATION_STD"]
    new_model = NeuralNet(hidden_size=SETTINGS["HIDDEN_SIZE"])
    new_model.load_state_dict(model.state_dict())
    
    with torch.no_grad():
        for param in new_model.parameters():
            # Randomly select mutation_rate% of weights to mutate
            mask = (torch.rand_like(param) < mutation_rate).float().to(DEVICE)
            noise = torch.randn_like(param).to(DEVICE) * std
            # Only add noise to selected weights
            param.add_(mask * noise)
            
    return new_model
