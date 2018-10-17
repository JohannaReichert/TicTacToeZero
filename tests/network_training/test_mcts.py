import unittest
import numpy as np
from sandbox import mcts
from source.game import tictactoe


#importlib.reload(mcts)


class TestSearch(unittest.TestCase):
    def setUp(self):
        self.mcts = mcts.MCTS()
        self.mcts.ttt = tictactoe.TTT()
        self.mcts.ttt.board = np.array(([1, -1, 0], # board with 3 available actions
                                   [-1, 0, 1],
                                   [1, -1, 0]))
        self.mcts.s = self.mcts.ttt.board.tostring() # converting ndarray to string so it can be used as key in a dict
        # creating sample Qsa, Psa and Nsa values to test if _select selects the action with maximal Q+U
        self.mcts.Qsa[(self.mcts.s,0)] = 1.5
        self.mcts.Qsa[(self.mcts.s,1)] = 2
        self.mcts.Qsa[(self.mcts.s,2)] = 3

        self.mcts.Psa[(self.mcts.s,0)] = 0.8
        self.mcts.Psa[(self.mcts.s,1)] = 0.0002
        self.mcts.Psa[(self.mcts.s,2)] = 0.01

        self.mcts.Nsa[(self.mcts.s,0)] = 2
        self.mcts.Nsa[(self.mcts.s,1)] = 3
        self.mcts.Nsa[(self.mcts.s,2)] = 1

        self.mcts.Ns[self.mcts.s] = 10
        c_puct = 1
        for action_nr in range(0,3): # calculating the Q+U values by hand to see which action has the largest Q+U
            q_u = self.mcts.Qsa[(self.mcts.s,action_nr)] + c_puct * self.mcts.Psa[(self.mcts.s,action_nr)] * np.sqrt(self.mcts.Ns[self.mcts.s]) / (
                    1 + self.mcts.Nsa[(self.mcts.s,action_nr)])
            print(f"Q+U for action {action_nr}: {q_u}")

    def test__select(self):
        print(len(self.mcts.ttt.show_available_moves()[0]))
        self.assertEqual(self.mcts._select(), 2)



if __name__ == '__main__':
    unittest.main()
