import pandas as pd
import numpy as np
import tkinter as tk
#from source.network_training import vanilla_mcts
import time


# 1 = x = random player or human, -1 = o = algo


class TTT():
    def __init__(self):
        self.board = np.full([3, 3], 0)

    # _def _create_new_board(self):
    #    self.board = np.full([3,3], 0)

    def _check_3inline(self, line):
        """checks whether all elements x in line are -1 or whether all elements are 1. Returns True if either
        of these conditions is true, and False otherwise"""
        return all(x == -1 for x in line) | all(x == 1 for x in line)

    def make_move(self,pos,players_turn):
        """applies a move to the current state of the board
        @pos: tuple of row, col index of move
        @player: -1 or 1, indicating which player applies the move"""
        if self.board[pos] == 0:
            self.board[pos] = players_turn
        else: print("Not possible! Position is not empty")

    def check_victory(self):
        """checks whether one of the players wins at the current state of board
        returns 1 if player_x wins, -1 if player_o wins, 0 if it is a draw"""

        # checking rows
        for x in range(3):
            if self._check_3inline(self.board[x, :]):
                return self.board[x,0]

        # checking cols
        for x in range(3):
            if self._check_3inline(self.board[:, x]):
                return self.board[0,x]

        # checking diagonals
        if self._check_3inline(np.diag(self.board)):
            return self.board[0,0]
        if self._check_3inline(np.diag(np.fliplr(self.board))):
            return self.board[0,2]

        return 0 #draw (or game not finished yet)

    def show_available_moves(self,board=None):
        """shows all possible legal moves (indices of board) at a current state of the board, i.e. all positions which are still empty
        returns tuple of an array with row indices and one with col indices"""
        if board is None:
            board = self.board
        moves = np.where(board==0)
        return moves # tuple of ndarrays

    def random_player(self):
        """randomly selects a move from all legal moves at the current state of board. returns the move as a tuple (row_index,col_index)"""
        moves_rows,moves_cols = self.show_available_moves()
        index = np.random.choice(moves_rows.shape[0], replace=False)
        pos = (moves_rows[index],moves_cols[index])
        return pos

    def _translate_idx(self,best_action):
        """takes best_action value returned by vanilla_mcts and converts it to a position in the 3x3 np.array"""
        range_array = np.arange(0,9).reshape((3,3))
        pos = np.where(range_array == best_action)
        return pos


    def mcts_player(self,players_turn):
        """selects a move based on the result of mcts and returns the move as a tuple (row_index,col_index)"""
        moves_rows,moves_cols = self.show_available_moves()
        if players_turn == 1:
            player = 'x'
        elif players_turn == -1:
            player = 'o'
        vmcts = vanilla_mcts.VanillaMCTS(n_iterations=500, depth=10, exploration_constant=1.4, game_board = self.board,tree = None, win_mark=3, player=player,log=False, fig = False)
        best_action, best_n, best_q, depth_searched = vmcts.solve()
        pos = self._translate_idx(best_action)
        return pos

    def play_game(self,player_x, player_o, log = False):
        """runs a game of tictactoe until one of the players wins or it is a draw.
        @player0: "random" --> random moves are picked. other options will be added
        @player1: "random" --> random moves are picked
        @log: determines whether the state of the board will be printed to the console or not
        returns 0 if player0 won, 1 if player1 won and -1 if it is a draw"""
        #self._create_new_board()

        players_turn = np.random.choice([-1,1]) # indicates whose turn it is (-1 or 1)

        while True:
            if log:
                print(f"Current board: \n {self.board}")
            available_moves = self.show_available_moves()
            if len(available_moves[0]) == 0:
                if log:
                    print("Game ended in a draw!")
                return 0

            if log:
                print(f" It is player {players_turn}'s turn now.")

            if players_turn == -1:
                if player_o == "random":
                    self.make_move(self.random_player(), -1)
                elif player_o == "mcts":
                    self.make_move(self.mcts_player(players_turn), -1)
            elif players_turn ==1:
                if player_x == "random":
                    self.make_move(self.random_player(),1)
                elif player_x == "mcts":
                    self.make_move(self.mcts_player(players_turn), 1)

            winner = self.check_victory()
            if winner != 0:
                if log:
                    print(f"Player {players_turn} wins!")
                    print(f"Winning board: \n {self.board}")
                return winner

            players_turn = -players_turn # if it was -1, now it is 1 (and vice versa) --> alternates between -1 and 1


if __name__ == "__main__":

    # watching them play
    '''
    ttt = TTT()
    ttt.play_game("mcts", "mcts", log = True)
    '''


    # getting some stats

    t0 = time.time()
    winner_list = []
    n_games = 100
    for i in range(n_games):
        ttt = TTT()
        winner_list.append(ttt.play_game("mcts", "random", log = False))

    t1 = time.time()


    """n_iterations=2000, depth=10, exploration_constant=1.4, game_board = self.board,tree = None, win_mark=3, player=player,log=False, fig = False)"""
    # mcts vs. mcts: 100 draws! 400 seconds
    # mcts vs. random: 96 vs. 0, 4 draws. 143 sec
    # 8000 iterations: mcts vs random: 93 vs 0, 7 draws. 492 sec
    # 500 iterations: mcts vs random 96 vs 0, 4 draws. 43 sec

    print(f"Percentage won by player_x: {(winner_list.count(1)/n_games)*100}%")
    print(f"Percentage won by player_o: {(winner_list.count(-1)/n_games)*100}%")
    print(f"Percentage draws: {(winner_list.count(0)/n_games)*100}%")
    print(f"Time taken for {n_games}: {t1-t0} s")









    """stats from running with mistake 3 (only updating levels of one player)"""
    """n_iterations=1000, depth=10, exploration_constant=1.4, game_board = self.board,tree = None, win_mark=3, player=player,log=False, fig = False"""
    # mcts vs. mcts: .39 vs. .37, .24 draws. 142 seconds
    # 2000 iterations: mcts vs. mcts: .49 vs. .45, .06 draws. 424 seconds

    """stats from running with mistake 2 (updating all levels) in simulation"""
    """mcts settings: n_iterations=1000, depth=20, exploration_constant=1.4, game_board = self.board,tree = None, win_mark=3, player=player,log=False, fig = False)"""
    # ramdomly chosen who starts
    # mcts vs. mcts: 0.4 vs 0.6, 0 draws. Time approx. 159.8
    """n_iterations = 3000, depth = 10, exploration_constant = 30, game_board = self.board, tree = None, win_mark = 3, player = player, log = False, fig = False)"""
    # mcts vs. mcts:
    # .48 vs 0.52. 459 sec

    """stats from running with mistake in simulation"""
    # mcts settings: n_iterations=1000, depth=20, exploration_constant=1.4, game_board = self.board, tree = None, win_mark=3, player=player,log=False
    # in all cases, player_o started! (therefore has advantage)
    # mcts vs. random: 0.89 vs. 0.1. Time approx 68 sec.
    # mcts vs. mcts: 0.41 vs. 0.59. Time approx 145 sec.
    # random vs. random: 0.25 vs. 0.65. Time approx 0.029 sec.

    # values if it was randomly chosen who starts:
    # mcts vs. random: 0.95 vs. 0.03. Time approx 71 sec.
    # mcts vs. mcts: 0.45 vs. 0.55 (0 draws). Time approx 140 sec.
    # random vs. random: 0.41 vs. 0.46 (0.13 draws). Time approx 0.032 sec.
    # if n_iter = 5000: still mcts vs. mcts = 0.59 vs. 0.41, 0 draws. Time 550 sec

    # higher exploration constant -> more draws (mcts vs. mcts approx. 0.2 draws)

