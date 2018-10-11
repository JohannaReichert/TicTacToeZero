import pandas as pd
import numpy as np
import tkinter as tk


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
        returns 0 if player 0 wins, 1 if player 1 wins, -1 if it is a draw"""

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

    def mcts_player(self):
        """randomly selects a move from all legal moves at the current state of board. returns the move as a tuple (row_index,col_index)"""
        moves_rows,moves_cols = self.show_available_moves()
        index = np.random.choice(moves_rows.shape[0], replace=False)
        pos = (moves_rows[index],moves_cols[index])
        return pos

    def play_game(self,player_x, player_o, log = False):
        """runs a game of tictactoe until one of the players wins or it is a draw.
        @player0: "random" --> random moves are picked. other options will be added
        @player1: "random" --> random moves are picked
        @log: determines whether the state of the board will be printed to the console or not
        returns 0 if player0 won, 1 if player1 won and -1 if it is a draw"""
        #self._create_new_board()

        players_turn = -1 # indicates whose turn it is (0 or 1)

        while True:
            if log:
                print(f"Current board: \n {self.board}")
            available_moves = self.show_available_moves()
            if len(available_moves[0]) == 0:
                if log:
                    print("Game ended in a draw!")
                return -1

            if log:
                print(f" It is player {players_turn}'s turn now.")

            if players_turn == -1:
                if player_o == "random":
                    self.make_move(self.random_player(), -1)
                elif player_o == "mcts":
                    self.make_move(self.mcts_player(), -1)
            elif players_turn ==1:
                if player_x == "random":
                    self.make_move(self.random_player(),1)

            winner = self.check_victory()
            if winner != 0:
                if log:
                    print(f"Player {players_turn} wins!")
                    print(f"Winning board: \n {self.board}")
                return winner

            players_turn = -players_turn # if it was -1, now it is 1 (and vice versa) --> alternates between -1 and 1


if __name__ == "__main__":
    ttt = TTT()
    ttt.play_game("random", "random", log = True)


