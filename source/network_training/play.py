import numpy as np
from source.game import tictactoe
from source.network_training import alphazero_mcts
from source.network_training import pure_mcts
from source.network_training import neural_network
from collections import defaultdict, deque
import random
import os
import time


def get_stats(game, player_x, player_o, alpha_mcts_solve_fn, pure_mcts_solve_fn, n_games=30, log = True):
    """
    in:
        @player_x, player_o: "random", "mcts" or "alphazero
        @alpha_mcts_solve_fn, pure_mcts_solve_fn: function "solve" from alphazero/mcts
        @n_games: number of games to be played for getting stats
        @log: boolean, if True games are printed to the console
    out: win_ratio of player_x (between 0 and 1)
    """
    t0 = time.time()
    winner_list = []
    n_games = n_games
    for i in range(n_games):
        if (i+1)%10 ==0: print(f"---------Game {i+1}----------------")
        winner_list.append(game.play_game(player_x, player_o, alpha_mcts_solve_fn, pure_mcts_solve_fn, log=log))

    print(f"Num_games:{n_games}, win player x ({player_x}): {(winner_list.count(1)/n_games)*100}%, "
          f"lose: {(winner_list.count(-1)/n_games)*100}%, draw:{(winner_list.count(0)/n_games)*100}%")
    win_ratio = winner_list.count(1) / n_games
    t1 = time.time()
    print(f"Time taken for {n_games}: {t1-t0} s")
    return win_ratio



if __name__ == "__main__":

    ttt = tictactoe.TTT()
    eval_net = neural_network.EvaluationNet('../saved_models/best_model.pt')
    eval_net.net_module.eval()

    alpha_mcts = alphazero_mcts.AlphaZeroMCTS(eval_net.evaluate_state_fn, n_iterations=2000, depth=10, exploration_constant=1.4,
                                              temperature = 0.1, game_board = ttt.board, tree = None, win_mark=3,
                                              player=None, log=False, fig = False)

    pure_mcts = pure_mcts.PureMCTS(n_iterations=5000, depth=10, exploration_constant=1.4, game_board = ttt.board,
                                   tree = None, win_mark=3, player=None, log=False, fig = False)

    #ttt.board = np.array(([-1, 0, 0],
    #                      [0, 1, 0],
    #                      [-1, 0, 1]))

    #best_action, best_n, best_q, depth_searched,move_probs,tree = alpha_mcts.solve(ttt.board,'x')
    #print('best action= ', best_action, ' best_n= ', best_n, ' best_q= ', best_q, 'depth_searched=', depth_searched,
    #      'move_probs=',move_probs,'tree=',tree)


    ttt.play_game(player_x = "alphazero", player_o = "mcts", alpha_mcts_solve_fn = alpha_mcts.solve,
                  pure_mcts_solve_fn = pure_mcts.solve, log = True)

    #get_stats(game = ttt, player_x = "alphazero", player_o = "mcts", alpha_mcts_solve_fn = alpha_mcts.solve,
    #          pure_mcts_solve_fn = pure_mcts.solve, n_games = 30, log = False)
