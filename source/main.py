
from source.game.tictactoe import TTT
from source.network_training import neural_network
from sandbox import mcts
#from source.network_training.neural_network import NeuralNet
import numpy as np

if __name__=="__main__":

    ttt = TTT()
    ttt.board = np.array(([1, -1, 0],
                          [-1, 0, 1],
                          [1, 1, 0]))
    nn = neural_network.NeuralNet(ttt)
    m = mcts.MCTS(ttt, nn)
    s = ttt.board.tostring()
    m._select(s)


    #t = training.Train(ttt, nn)

    #t.self_play()

    #t.learn()


