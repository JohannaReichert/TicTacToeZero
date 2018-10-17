import numpy as np
from sandbox import mcts


class Train():
    """combining the mcts and neural network functions to train via self-play"""
    def __init__(self,game,neural_net):
        self.game = game # instance of ttt
        self.neural_net = neural_net
        self.mcts = mcts.MCTS(self.game, self.neural_net)
        self.temp_threshold = 4

    def _choose_action(self,board,episode_step,available_moves):
        temperature = int(episode_step < self.temp_threshold)
        probs = self.mcts.get_move_probs(board, available_moves, temp=temperature) # vector of probabilities for each possible action
        action = np.random.choice(len(probs), p=probs)  # selects an action from the possible actions. actions are represented
        # according to their probabilites (e.g. if an action has prob = 0.5 and there are 8 actions, this action will be present 4 times).
        # len(probs) is the nr of possible actions. action will be an int indicating the index of the action in the order of probs
        pos = available_moves[action]
        return pos # should be tuple of row, col index of move

    def self_play(self,log=True):
        """runs one episode of self-play, in which each move of player0 is decided via mcts simulations
        It uses a temperature =1 if episodeStep < temp_threshold, and thereafter uses temp=0"""
        board = self.game.board
        episode_step = 0
        players_turn = 0  # indicates whose turn it is (0 or 1)
        while True:
            episode_step += 1
            if log:
                print(f"Current board: \n {self.game.board}")
            available_moves = self.game.show_available_moves()
            if len(available_moves[0]) == 0:
                if log:
                    print("Game ended in a draw!")
                return -1

            if log:
                print(f" It is player {players_turn}'s turn now.")


            if players_turn == -1:
                self.game.make_move(self.game.mcts_player(players_turn), -1)
            elif players_turn ==1:
                self.game.make_move(self.game.mcts_player(players_turn), 1)


            winner = self.game.check_victory()
            if winner != -1:
                if log:
                    print(f"Player {players_turn} wins!")
                    print(f"Winning board: \n {self.board}")
                return winner

            players_turn = 1 -players_turn # if it was 0, now it is 1 (and vice versa) --> alternates between 0 and 1


    def learn(self):
        pass
