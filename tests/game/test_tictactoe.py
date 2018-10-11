import unittest
from source.game import tictactoe
import numpy as np


class TestCheck_victory(unittest.TestCase):
    def setUp(self):
        self.ttt = tictactoe.TTT()

    def test_check_draw(self):
        """checking whether 0 is returned in case of a draw"""
        self.ttt.board = np.array(([1, 1, -1],
                                    [-1, -1, 1],
                                    [1, 1, -1]))
        self.assertEqual(self.ttt.check_victory(), 0)

    def test_check_1win(self):
        """checking whether 1 is returned if player_x wins"""
        self.ttt.board = np.array(([1, 1, -1],
                                    [0, 1, 1],
                                    [1, 0, 1]))
        self.assertEqual(self.ttt.check_victory(), 1)

    def test_check_0win(self):
        """checking whether -1 is returned if player_o wins"""
        self.ttt.board = np.array(([1, 1, -1],
                                    [0, -1, 1],
                                    [-1, 0, 1]))
        self.assertEqual(self.ttt.check_victory(), -1)

class Test__check_3inline(unittest.TestCase):
    def setUp(self):
        self.ttt = tictactoe.TTT()
        self.line_mixed = [0,1,-1,0]
        self.line_allneg1 = [-1,-1,-1,-1,-1]
        self.line_all1 = [1,1,1]

    def test_3inline_returns(self):
        """checking whether the _check_a3inline returns True if all elements are -1 or 1 and False otherwise"""
        self.assertEqual(self.ttt._check_3inline(self.line_mixed),False)
        self.assertEqual(self.ttt._check_3inline(self.line_allneg1), True)
        self.assertEqual(self.ttt._check_3inline(self.line_all1), True)


class Test_show_available_moves(unittest.TestCase):
    def setUp(self):
        self.ttt = tictactoe.TTT()

    def test_show_available_moves(self):
        """checking whether the indices of empty positions are returned correctly"""
        self.ttt.board = np.array(([1, 0, -1],
                                    [0, -1, 1],
                                    [1, 1, -1]))
        np.testing.assert_array_equal(self.ttt.show_available_moves(),(np.array(([0,1])),np.array(([1,0]))))


        self.ttt.board = np.array(([0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0]))
        np.testing.assert_array_equal(self.ttt.show_available_moves(),(np.array(([0, 0, 0, 1, 1, 1, 2, 2, 2])),
                                                                                     np.array(([0, 1, 2, 0, 1, 2, 0, 1, 2]))))
class Test_make_move(unittest.TestCase):
    def setUp(self):
        self.ttt = tictactoe.TTT()
        self.ttt.board =  np.array(([1, 0, -1],
                                    [0, -1, 1],
                                    [1, 1, -1]))

    def test_make_move(self):
        """checking whether applying a move at a certain position changes the board accordingly"""
        self.ttt.make_move((0, 1), -1)
        np.testing.assert_array_equal(self.ttt.board,np.array(([1, -1, -1],
                                                               [0, -1, 1],
                                                               [1, 1, -1])))

        # making sure moving to a non-empty position will not change the board
        # "Not possible! Position is not empty" should be printed to console
        self.ttt.make_move((0, 1), 1)
        np.testing.assert_array_equal(self.ttt.board,np.array(([1, -1, -1],
                                                               [0, -1, 1],
                                                               [1, 1, -1])))

class Test_random_player(unittest.TestCase):
    def setUp(self):
        self.ttt = tictactoe.TTT()
        #self.ttt._create_new_board()

    def test_randomplayer(self):
        """testing whether the board is filled correctly with 1s if player 1 plays 9 times"""
        self.ttt.make_move(self.ttt.random_player(), 1)
        self.ttt.make_move(self.ttt.random_player(), 1)
        self.ttt.make_move(self.ttt.random_player(), 1)
        self.ttt.make_move(self.ttt.random_player(), 1)
        self.ttt.make_move(self.ttt.random_player(), 1)
        self.ttt.make_move(self.ttt.random_player(), 1)
        self.ttt.make_move(self.ttt.random_player(), 1)
        self.ttt.make_move(self.ttt.random_player(), 1)
        self.ttt.make_move(self.ttt.random_player(), 1)
        self.assertEqual(len(self.ttt.show_available_moves()[0]),0)
        np.testing.assert_array_equal(self.ttt.board,np.array(([1, 1, 1],
                                                               [1, 1, 1],
                                                               [1, 1, 1])))


if __name__ == "__main__":
    unittest.main()