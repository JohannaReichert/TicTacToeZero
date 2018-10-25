import numpy as np
from source.game import tictactoe
from source.network_training import alphazero_mcts
from source.network_training import neural_network
from collections import deque
import random
import os
import time

class Train():
    """combining the mcts and neural network functions to train via self-play"""
    def __init__(self, saved_model = None):
        self.ttt = tictactoe.TTT() # instance of ttt
        self.temp_threshold = 4
        self.learning_rate = 0.01 #2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature parameter
        self.n_iter = 500  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 256 # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50

        self.game_batch_num = 1500 #1500

        self.target_win_ratio = 0.55

        if saved_model:
            print("LOADING A SAVED MODEL")
            # start training from a previously saved neural network
            self.eval_net = neural_network.EvaluationNet(saved_model)
        else:
            # start training from a new network
            self.eval_net = neural_network.EvaluationNet()

        self.alpha_mcts = alphazero_mcts.AlphaZeroMCTS(self.eval_net.evaluate_state_fn, n_iterations=self.n_iter, depth=10,
                                                       exploration_constant=1.4, temperature = self.temp, tree=None, win_mark=3, game_board=self.ttt.board,
                                                       player=None, log=False, fig=False)


    def _choose_action(self,board,episode_step,available_moves):
        temperature = int(episode_step < self.temp_threshold)
        probs = self.mcts.get_move_probs(board, available_moves, temp=temperature) # vector of probabilities for each possible action
        action = np.random.choice(len(probs), p=probs)  # selects an action from the possible actions. actions are represented
        # according to their probabilites (e.g. if an action has prob = 0.5 and there are 8 actions, this action will be present 4 times).
        # len(probs) is the nr of possible actions. action will be an int indicating the index of the action in the order of probs
        pos = available_moves[action]
        return pos # should be tuple of row, col index of move

    def get_rotation_data(self):
        """might be implemented later: using rotations of the board to increase stats"""

    def store_self_play_data(self, n_games = 1):
        for i in range(n_games):
            play_data = self.ttt.self_play(self.alpha_mcts.solve,log = False)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            self.data_buffer.extend(play_data)
            #print(self.episode_len)
            #print(self.data_buffer)

    def optimize_evaluation_net(self):
        """optimise the weights of evaluation net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        states_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winners_batch = [data[2] for data in mini_batch]
        next_players_batch = [data[3] for data in mini_batch]
        old_probs, old_value = self.eval_net.evaluate_batch(states_batch)
        for i in range(self.epochs):
            loss, entropy = self.eval_net.train_step(
                    states_batch,
                    mcts_probs_batch,
                    winners_batch,
                    self.learning_rate*self.lr_multiplier)
            new_probs, new_value= self.eval_net.evaluate_batch(states_batch)
##check
        """
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        """

        explained_var_old = (1 -
                             np.var(np.array(winners_batch) - old_value.flatten()) /
                             np.var(np.array(winners_batch))+.00000000001)
        explained_var_new = (1 -
                             np.var(np.array(winners_batch) - new_value.flatten()) /
                             np.var(np.array(winners_batch))+.00000000001)
        print((#"kl:{:.5f},"
               #"lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(#kl,
                        #self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def test_net_against_player(self, n_games=30, player="mcts"):
        """
        Evaluate the trained policy by playing against the pure MCTS player or random player
        Note: this is only for monitoring the progress of training
        """
        tic = tictactoe.TTT()
        t0 = time.time()
        winner_list = []
        n_games = n_games
        for i in range(n_games):

            print(f"---------Game {i+1}----------------")
            winner_list.append(tic.play_game("alphazero", player, log=True))

        print(f"Num_games:{n_games}, win az: {(winner_list.count(1)/n_games)*100}%, "
              f"lose: {(winner_list.count(-1)/n_games)*100}%, draw:{(winner_list.count(0)/n_games)*100}%")
        win_ratio = winner_list.count(1) / n_games
        t1 = time.time()
        print(f"Time taken for {n_games}: {t1-t0} s")
        return win_ratio

    def test_net_against_previous(self, n_games = 30, previous_best_model_path = None):
        """Evaluate the current alphazero model by playing against the last stored alphazero model"""
        tic = tictactoe.TTT()
        eval_net_old = neural_network.EvaluationNet(previous_best_model_path)
        current_alphazero = alphazero_mcts.AlphaZeroMCTS(self.eval_net.evaluate_state_fn, n_iterations=self.n_iter, depth=10,
        exploration_constant=1.4, temperature = 1, tree=None, win_mark=3, game_board=tic.board, player=None, log=False, fig=False)

        previous_alphazero = alphazero_mcts.AlphaZeroMCTS(eval_net_old.evaluate_state_fn, n_iterations=self.n_iter, depth=10,
        exploration_constant=1.4, temperature = 1, tree=None, win_mark=3, game_board=tic.board, player=None, log=False, fig=False)
        winner_list = []
        n_games = n_games
        for i in range(n_games):
            tic = tictactoe.TTT()
            winner_list.append(tic.play_game("alphazero", "alphazero_old", current_alphazero.solve, None,
                                             prev_alpha_solve_fn= previous_alphazero.solve, log=False))

        print(f"Num_games:{n_games}, win alphazero new: {(winner_list.count(1)/n_games)*100}, "
              f"lose: {(winner_list.count(-1)/n_games)*100}, draw:{(winner_list.count(0)/n_games)*100}")
        win_ratio = winner_list.count(1)/n_games

        return win_ratio


    def run_training(self):
        """run the training pipeline"""
        try:
            loss =-1
            entropy = -1
            for i in range(self.game_batch_num):
                self.store_self_play_data(self.play_batch_size)
                print(f"batch i: {i+1}, episode_len: {self.episode_len}")
                if len(self.data_buffer) > self.batch_size: # if we have accumulated enough data to optimize the net
                    loss, entropy = self.optimize_evaluation_net()
                previous_best_model_path = '../saved_models/best_model.pt'
                if i == 0 and not os.path.exists(previous_best_model_path):
                    self.eval_net.save_model(previous_best_model_path)
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    print(f"current self-play batch: {i+1}")
                    print("loss:",loss)
                    print("entropy:",entropy)
                    current_model_path = '../saved_models/current_model.pt'
                    self.eval_net.save_model(current_model_path)
                    self.eval_net.net_module.eval() # switch to evaluation mode of model
                    win_ratio = self.test_net_against_previous(n_games=30, previous_best_model_path = previous_best_model_path)
                    if win_ratio > self.target_win_ratio:
                        print("New best prediction!!!")
                        # update the best model
                        self.eval_net.save_model('../saved_models/best_model.pt')

        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == "__main__":


    #t = Train('../saved_models/best_model.pt')
    t = Train()

    #t.run_training()



    # nach 150 self_play spielen mit je 500 iterationen: schon nur draws gegen mcts, 90% gewonnen gegen random (7% draws)

    # i450 new best prediction 60 % win. (vorheriges best model war mit bugs trainiert)