import pandas as pd
import numpy as np
import tkinter as tk
#from source.network_training import pure_mcts
#from source.network_training import alphazero_mcts
#from source.network_training import neural_network
import time
import copy


# 1 = x = random player or human, -1 = o = algo


class TTT():
    def __init__(self):
        self.board = np.full([3, 3], 0)
        self.players_turn = -99 # will tell whose turn it is (1 for x and -1 for o)

    def _create_new_board(self):\
        self.board = np.full([3,3], 0)

    def _check_3inline(self, line):
        """checks whether all elements x in line are -1 or whether all elements are 1. Returns True if either
        of these conditions is true, and False otherwise"""
        return all(x == -1 for x in line) | all(x == 1 for x in line)

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
        returns tuple of an array with row indices and one with col indices, e.g. ([0,2,1],[0,1,2])"""
        if board is None:
            board = self.board
        moves = np.where(board==0)
        return moves # tuple of ndarrays

    def _translate_idx(self,best_action):
        """takes best_action value returned by pure_mcts or alphazero_mcts and converts it to a position in the 3x3 np.array"""
        range_array = np.arange(0,9).reshape((3,3))
        pos = np.where(range_array == best_action)
        return pos

    def make_move(self,pos):
        """applies a move to the current board
        @pos: tuple of row, col index of move: (row,col)
        players_turn = -1 or 1, depending on which player applies the move"""
        if self.board[pos] == 0:
            self.board[pos] = self.players_turn
        else: print("Not possible! Position is not empty")


    def create_board_state(self, board, players_turn):
        """uses information about the next player (i.e. the player whose turn it is) and the board to create a combined
         board state of one array representing the current board, and one representing the player (full of -1s or 1s)"""
        board_state = np.zeros((2, board.shape[0], board.shape[1]))
        board_state[0] = board
        board_state[1][:, :] = players_turn

        reshaped_board = np.array(board_state.reshape(2, board.shape[0], board.shape[1]))

        return reshaped_board


    def decide_move(self, player_type, alpha_mcts_solve_fn, pure_mcts_solve_fn, prev_alpha_solve_fn):
        """decides the position (pos) on the board which will be filled with -1 or 1 (depending on self.players_turn,
        depending on player_type, which can be "random", "mcts", "alphazero" or "alphazero_old"""
        if self.players_turn == 1:
            player_symbol = 'x'
        elif self.players_turn == -1:
            player_symbol = 'o'

        if player_type == "random":
            moves_rows, moves_cols = self.show_available_moves()
            index = np.random.choice(moves_rows.shape[0], replace=False)
            pos = (moves_rows[index], moves_cols[index])
        elif player_type == "mcts":
            best_action, best_n, best_q, depth_searched = pure_mcts_solve_fn(self.board, player_symbol)
            pos = self._translate_idx(best_action)
        elif player_type == "alphazero":
            best_action, best_n, best_q, depth_searched, probs, tree = alpha_mcts_solve_fn(self.board, player_symbol)
            pos = self._translate_idx(best_action)
        elif player_type == "alphazero_old":
            best_action, best_n, best_q, depth_searched, probs, tree = prev_alpha_solve_fn(self.board, player_symbol)
            pos = self._translate_idx(best_action)

        return pos


    def play_game(self, player_x, player_o, alpha_mcts_solve_fn, pure_mcts_solve_fn, prev_alpha_solve_fn = None, log = False):
        """runs a game of tictactoe until one of the players wins or it is a draw.
        @player_x and player_o: "random", "mcts", "alphazero" or "alphazero_old"
        @log: determines whether the state of the board will be printed to the console or not
        returns 1 if player_x won, -1 if player_o won and 0 if it was a draw"""

        self._create_new_board()
        self.players_turn = np.random.choice([-1,1]) # indicates whose turn it is (-1 or 1)

        while True:

            if log:
                print(f"Current board: \n{self.board}")

            available_moves = self.show_available_moves()

            if len(available_moves[0]) == 0:
                if log:
                    print("Game ended in a draw!")
                return 0

            if log:
                print(f" It is player {self.players_turn}'s turn now.")
            if self.players_turn ==1:
                self.make_move(self.decide_move(player_x, alpha_mcts_solve_fn, pure_mcts_solve_fn, prev_alpha_solve_fn))
            elif self.players_turn ==-1:
                self.make_move(self.decide_move(player_o, alpha_mcts_solve_fn, pure_mcts_solve_fn, prev_alpha_solve_fn))

            winner = self.check_victory()
            if winner != 0:
                if log:
                    print(f"Player {self.players_turn} wins!")
                    print(f"Winning board: \n {self.board}")
                return winner

            self.players_turn = -self.players_turn # alternates between -1 and 1

    def self_play(self, alpha_mcts_solve_fn, log = False, temperature=.1):
        """ do a self-play game using the alphazero MCTS player, reuse the search tree, and store the self-play data:
            (state, mcts_probs, winner_z) for training
        in: board,
        out: zipped lists of np arrays of (states, mcts_probs, winners)
            each item of states is a 3x3 board, each of mcts_probs is a 3x3 array of probs, each of winners is a 1-dimensional np.array of length
            n_turns in the game
        """
        self._create_new_board()
        n_turns = 0
        states, mcts_probs, next_players = [], [], [] # next_players indicates whose turn it is at current state of board, -1 or 1
        self.players_turn = np.random.choice([-1, 1])  # players_turn indicates whose turn it is (-1 or 1)
        tree = None

        while True:
            if self.players_turn == 1:
                player_symbol = 'x'
            elif self.players_turn == -1:
                player_symbol = 'o'

            if log:
                print(f"Current board: \n {self.board}")

            available_moves = self.show_available_moves()

            if len(available_moves[0]) == 0: # no more moves left --> must be a draw (otherwise would have exited below at check_victory()
                winner = 0
                winners_z = np.zeros(len(next_players))
                if log:
                    print("Game ended in a draw!")
                break
            if log:
                print(f" It is player {self.players_turn}'s turn now.")

            # getting move and probs from alphazero mcts (which gets them from neural net)
            if n_turns<5:
                best_action, _, _, _, move_probs, _ = alpha_mcts_solve_fn(self.board, player_symbol,self_play=True)
            else: # temp is decreased after 5 turns
                best_action, _, _, _, move_probs, _ = alpha_mcts_solve_fn(self.board, player_symbol,temperature = 0.1,self_play=True)

            pos = self._translate_idx(best_action)

            # collecting data
            board_state = self.create_board_state(copy.deepcopy(self.board),copy.deepcopy(self.players_turn))
            states.append(board_state)
            mcts_probs.append(move_probs)
            next_players.append(copy.deepcopy(self.players_turn))

            self.make_move(pos)
            n_turns += 1 # increase n_turns +1

            winner = self.check_victory()
            if winner != 0:
                winners_z = np.zeros(len(next_players))
                winners_z[np.array(next_players) == winner] = 1.0
                winners_z[np.array(next_players) != winner] = -1.0
                if log:
                    print(f"Player {self.players_turn} wins!")
                    print(f"Winning board: \n {self.board}")
                break


            self.players_turn = -self.players_turn  # alternates between -1 and 1

        return zip(states, mcts_probs, winners_z, next_players)


if __name__ == "__main__":



    """n_iterations=2000, depth=10, exploration_constant=1.4, game_board = self.board,tree = None, win_mark=3, player=player,log=False, fig = False)"""
    """
    # mcts vs. mcts: 100 draws! 400 seconds
    # mcts vs. random: 96 vs. 0, 4 draws. 143 sec
    # 8000 iterations: mcts vs random: 93 vs 0, 7 draws. 492 sec
    # 500 iterations: mcts vs random 96 vs 0, 4 draws. 43 sec
    # 500 iterations: mcts vs. mcts: 95% draws. 107 sec
    """






