import numpy as np
import gym
from gym import spaces

class ReferentialGame(gym.Env):
    def __init__(self, vocab_size=5, num_objects=3):
        super().__init__()
        self.vocab_size = vocab_size  # Number of unique words in the language
        self.num_objects = num_objects  # Number of objects to choose from

        # Speaker sees one object (Discrete)
        self.observation_space = spaces.Discrete(num_objects)  

        # Speaker selects a "word" from the vocabulary (Discrete)
        self.action_space = spaces.Discrete(vocab_size)  

        # Listener selects an object (Discrete)
        self.listener_action_space = spaces.Discrete(num_objects)  

    def reset(self):
        self.target_object = np.random.randint(0, self.num_objects)  # Random object selection
        return self.target_object  # Speaker receives the target object as input

    def step(self, speaker_action, listener_action):
        """
        speaker_action: A message from the vocabulary.
        listener_action: The guessed object from the listener.
        """
        reward = 1.0 if listener_action == self.target_object else 0.0  # Correct guess = reward 1
        return self.target_object, reward, True, {}  # Single-step episode
