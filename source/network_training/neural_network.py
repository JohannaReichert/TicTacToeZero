from source.game import tictactoe
#from source.network_training import mcts
import numpy as np

class NeuralNet():
    def __init__(self,game):
        self.game = game
        pass


    def predict_probs(self):
        """for now, just return random vals
        outputs action probabilities for the game state of the leaf node

        """

        action_len = len(self.game.show_available_moves()[0])
        return np.random.rand(action_len)


    def predict_value(self):
        """for now, just return a random val
        outputs value of the state for the current player"""
        return np.random.rand(1)*10

    def train(self):
        pass
