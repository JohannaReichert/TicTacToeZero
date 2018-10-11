import numpy as np
import math
from source.game.tictactoe import TTT
from source.network_training.neural_network import NeuralNet

class MCTS():
    """class for implementation of monte-carlo tree search
    for each board state, n_epis mcts are run (for Google's AlphaZero, n_epis was 1600. For tictactoe, here 10 was chosen
    as the search state is much smaller"""

    def __init__(self,game,neural_net):
        #, neural_net
        """each node s in the search tree contains edges s,a for all legal actions. Each of these edges stores a set of statistics:
        N(s,a), W(s,a), Q(s,a), P(s,a)"""

        self.neural_net = neural_net
        self.game = game

        self.Nsa = {} # visit count of edge s,a
        self.Wsa = {} # total action-value of edge s,a
        self.Qsa = {} # mean action-value of edge s,a
        self.Ps = {} # prior probabilities of the actions possible at state s

        self.Ns = {}

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s
        #self.ttt = TTT()
        #self.s = self.ttt.board.tostring()

        self.n_epis = 10

    def _select(self, s, c_puct = 1):
        """looks at all possible actions at board state s and selects the action with maximal Q + U value (upper confidence bound).
        Returns the index of this action in the row_array and col_array of show_available_moves()
        cpuct = hyperparameter that controls the degree of exploration"""
        possible_moves = self.game.show_available_moves() # show_available_moves() is a tuple of (row_array, col_array)
        print(possible_moves)
        max_q_u, best_a = -float("inf"), -1
        for action_nr in range(len(possible_moves[0])):
            q_u = self.Qsa[(s,action_nr)] + c_puct * self.Ps[(s,action_nr)] * np.sqrt(self.Ns[s]) / (
                    1 + self.Nsa[(s,action_nr)])
            if q_u > max_q_u:
                max_q_u = q_u
                best_a = action_nr
        a_idx = best_a
        return a_idx

    def _expand_evaluate(self):
        """when arrived at a leaf node, this node is added to a queue for neural network evaluation. Leaf node is then expanded and
        edge values initalized to 0"""


    def _backup(self):
        """the edge statistics are updated in a backward pass. Visit count Nsa = Nsa + 1. Wsa = Wsa + v. Qsa = Wsa/N"""
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1

    def search(self, s, neural_net):
        """uses _select, _expand_evaluate and _backup"""

        """
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(board, 1)
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]
        """
        if s not in self.Ps:
            # s is a leaf node
            self.Ps[s] = self.nnet.predict_probs()
            v = self.neural_net.predict_value()

            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize

            self.Ns[s] = 0


            return v


        a = self._select(s)


        v = self.search(next_s)


        return -v


    def get_move_probs(self,board,available_moves,temp):
        """returns vector of probabilities for the possible moves at the current board state"""
        s =  board.tostring()

        for i in range(self.n_epis):
            self.search(s,available_moves)



        # creating list of visit counts for the visited nodes
        ######## range must be changed!!!
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(999)]

        if temp==0:
            best_action = np.argmax(counts)
            probs = [0]*len(counts)
            probs[best_action]=1
            return probs

        # probs are proportional to exponentiated visit count (p. 26)
        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]

        return probs



    def play(self):
        """after n_epis of simulations, a  move a is selected to play in the root position. The move is selected based
        on its exponentiated visit count (see page 26). search tree is reused (child becomes new root node, subtree is
        retained, while remainder of tree is discarded"""


tic = TTT()
s = tic.board.tostring()
nn = NeuralNet(tic)
m = MCTS(tic,nn)
m._select(s)




