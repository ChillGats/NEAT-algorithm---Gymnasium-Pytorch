import torch
import torch.nn as nn
import random
from game import SETTINGS

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

    def forward(self, x):
        return self.net(x)

# === NOUVEAUX OPÉRATEURS GÉNÉTIQUES ===

def crossover(parent1, parent2):
    """Mélange les poids de deux réseaux de neurones (50/50)."""
    child = NeuralNet(hidden_size=SETTINGS["HIDDEN_SIZE"])
    state1 = parent1.state_dict()
    state2 = parent2.state_dict()
    child_state = {}
    
    with torch.no_grad():
        for key in state1:
            # Crée un masque aléatoire de 0 et 1 pour choisir quel parent donne quel neurone
            mask = torch.rand_like(state1[key]) > 0.5
            # Combine les gènes
            child_state[key] = torch.where(mask, state1[key], state2[key])
            
    child.load_state_dict(child_state)
    return child

def mutate(model, mutation_rate=0.2, std=None):
    """Ne mute que 20% des poids, pour ne pas détruire l'apprentissage."""
    std = std if std else SETTINGS["MUTATION_STD"]
    new_model = NeuralNet(hidden_size=SETTINGS["HIDDEN_SIZE"])
    new_model.load_state_dict(model.state_dict())
    
    with torch.no_grad():
        for param in new_model.parameters():
            # Choisit aléatoirement 20% des poids à muter
            mask = (torch.rand_like(param) < mutation_rate).float()
            noise = torch.randn_like(param) * std
            # N'ajoute le bruit qu'aux poids sélectionnés
            param.add_(mask * noise)
            
    return new_model
