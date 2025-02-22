import torch
import torch.nn as nn
import torch.optim as optim
import random

# Speaker Network: Maps an object to a word (discrete token)
class Speaker(nn.Module):
    def __init__(self, vocab_size, num_objects):
        super().__init__()
        self.fc = nn.Linear(num_objects, vocab_size)  # Object → Word

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)  # Predicts probability over vocab

# Listener Network: Maps a word back to an object
class Listener(nn.Module):
    def __init__(self, vocab_size, num_objects):
        super().__init__()
        self.fc = nn.Linear(vocab_size, num_objects)  # Word → Object

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)  # Predicts probability over objects
