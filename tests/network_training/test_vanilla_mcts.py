import unittest
from source.game import tictactoe
from source.network_training import vanilla_mcts
import numpy as np


class Test_selection(unittest.TestCase):
    def setUp(self):
        # setting up a board with 5 empty positions
        self.tic = tictactoe.TTT()
        self.tic.make_move((0, 1), -1)
        self.tic.make_move((1, 0), 1)
        self.tic.make_move((1, 1), -1)
        self.tic.make_move((2, 0), 1)
        self.vmcts = vanilla_mcts.VanillaMCTS(n_iterations=1, depth=2, exploration_constant=1.4, game_board = self.tic.board,tree = None, win_mark=3, player='x',log=True, fig = True)

        #self.vmcts.expansion((0,) ) # creating all possible actions (children) for this board state
        # setting q, w, n and total n values
        #vmcts.tree[0,]['w']


    def test_selects_best_action(self):
        """checks whether the action with maximal ucb value is selected and returned"""
        #self.vmcts.selection()


    def test__get_valid_actions(self):

        possible_actions = self.vmcts._get_valid_actions(self.tic.board)
        print(possible_actions)
        print(len(possible_actions))


class Test_simulation(unittest.TestCase):
    def setUp(self):
        self.tic = tictactoe.TTT()
        self.tic.board = np.array(([-1, 0, 0],
                                   [-1, 1, 0],
                                   [-1, 1, 1]))
        self.vmcts = vanilla_mcts.VanillaMCTS(n_iterations=10, depth=10, exploration_constant=1.4,
                                              game_board=self.tic.board, tree=None, win_mark=3, player='x', log=True,
                                              fig=True)
        self.ticdraw = tictactoe.TTT()
        self.ticdraw.board = np.array(([-1, -1, 1],
                                       [ 1, 1, -1],
                                       [-1, 1,  1]))
        self.vmctsdraw = vanilla_mcts.VanillaMCTS(n_iterations=10, depth=10, exploration_constant=1.4,
                                              game_board=self.ticdraw.board, tree=None, win_mark=3, player='x', log=True,
                                              fig=True)
        self.ticxwins = tictactoe.TTT()
        self.ticxwins.board = np.array(([0, -1, 1],
                                       [ 1, 1, -1],
                                       [-1, 1,  1]))
        self.vmctsxwins = vanilla_mcts.VanillaMCTS(n_iterations=10, depth=10, exploration_constant=1.4,
                                              game_board=self.ticxwins.board, tree=None, win_mark=3, player='x', log=True,
                                              fig=True)
        self.tico = tictactoe.TTT()
        self.tico.board = np.array(([-1, 1, 0],
                                     [0, 1, 1],
                                    [-1, -1, 1]))
        # its o's (-1s) turn
        self.vmctso = vanilla_mcts.VanillaMCTS(n_iterations=10, depth=10, exploration_constant=1.4,
                                              game_board=self.tico.board, tree=None, win_mark=3, player='o', log=False,
                                              fig=True)

    def test_simulation(self):
        """test whether in case of a terminated board the correct winner is returned"""
        self.assertEqual(self.vmcts.simulation((0,)),'o')
        self.assertEqual(self.vmctsdraw.simulation((0,)), 'draw')
        self.assertEqual(self.vmctsxwins.simulation((0,)), 'x')
        owinner = []
        nsim = 1000
        for sim in range(nsim):
            owinner.append(self.vmctso.simulation((0,)))
        print(f" o wins: {owinner.count('o')}")
        print(f" x wins: {owinner.count('x')}")

