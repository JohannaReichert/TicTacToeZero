import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from source.game.tictactoe import TTT
#from source.network_training import neural_network



####todo:

# warum wird alpha besser obwohl loss nicht runtergeht?
# change way saved_model can be added
# test 2 models against each other
# check reusing of tree
# regeln aus spiel ziehen
# implement human player GUI
# what does the nn "see"?
# check wording: "moves" vs. "actions", probs and pi etc.


class AlphaZeroMCTS(object):
    def __init__(self, evaluate_state_fn, n_iterations=50, depth=10, exploration_constant=1.4, temperature = 1,
                 tree=None, win_mark=3, game_board=None, player=None, log=False, fig=True):
        """@tree: nested dictionary, format see _set_tictactoe
        @ win_mark: if the row-wise, col-wise or diagonal-wise sum reaches this number, player 'x' (represented by 1 on the board)
            has won. If the sum = -win_mark, the player 'o' (using -1) has won
        @ player: player whose turn it is now. 'o' or 'x'
        @ log: boolean, if True infos will be printed to console
        @ fig: boolean, if True figure with stats will be shown"""
        self.temp = temperature
        self.evaluate_state_fn = evaluate_state_fn
        self.n_iterations = n_iterations
        self.depth = depth
        self.exploration_constant = exploration_constant
        self.total_n = 0
        self.max_depth_searched = 0
        self.root_id = (0,)
        self.game_board = game_board

        self.log = log
        self.fig = fig

        self.orig_player = player
        self.leaf_node_id = None
        self.n_rows = len(game_board)  # normal tictactoe: 3
        self.win_mark = win_mark  # later used to check victory (tictactoe: 3 in a row)

        if tree == None:
            self._set_tree(game_board, player)
        else:
            self.tree = tree
            if self.log:
                print("TREE IS REUSED")
###
            """
            # maybe don't do this whole paragraph??
            self.total_n = self.tree[(0,)]['n']
            #print("player:", player)
            #print("root player:", self.tree[(0,)]['next_player'])
            # if the tree was not built for the same original player, all signs of w and q have to be reversed
            # the tree root's next_player is the player who the tree was NOT built for --> if they are the same,
            # we have to change the signs

            if player == self.tree[(0,)]['next_player']:
                if self.log:
                    print("CHANGING SIGNS!!!")
                for node_id, node_info in self.tree.items():
                    node_info['w'] = - node_info['w']
                    node_info['q'] = - node_info['q']
            """

    def _set_player(self,player):
        self.orig_player = player
        self.tree[self.root_id]['next_player'] = self.orig_player

    def _set_board(self,board):
        self.game_board = board

    def _set_tree(self, game_board, player):
        """creates a nested tree dictionary with key = node_id, value = dict with keys state, player, child, parent, n, w, q.
        n, w and q will be filled later with: n = visit count of node, w = win count, q = w/n, p = probabilities for moves"""
        #print("setting tree")
        #print("player",player)
        self.total_n = 0
        self.leaf_node_id = None
        self.tree = {self.root_id: {'state': game_board,
                          'next_player': player,
                          'children': [],
                          'parent': None,
                          'n': 0,
                          'w': 0,
                          'q': None,
                          'p': 0}}



    def _update_tree(self, best_action):
        """after solving, update the tree so that the selected action is the new root. discard all the subtrees whose node_id
        doesn't start with the new root id"""
        new_root = self.root_id + (best_action,) # e.g. (0,3); the id of the selected action

        #print("tree before updating ids")

        #test= {k: self.tree[k] for k in list(self.tree.keys()) if
        #           k[0:len(new_root)] == new_root}
        #print(test)

        # for every node in the subtree of the selected action, replace the node id. e.g. (0,3,1) becomes (0,1)
        subtree = {(0,) + k[len(new_root):] : self.tree[k] for k in list(self.tree.keys()) if k[0:len(new_root)] == new_root}
        # update parent ids
        for node_id, node_info in subtree.items():
            old_parent_id = node_info['parent']
            node_info['parent'] = (0,) + old_parent_id[len(new_root):]

        subtree[(0,)]['parent'] = None # the new root does not have a parent

        self.tree = subtree


    def selection(self):
        """ iterates through the game tree by always selecting the action (child) with maximum ucb value.
        when the action is a leaf node, it is returned
        in: self.tree
        out:
            - leaf node id (node to expand)
            - depth (depth of node root=0)"""

        leaf_node_found = False
        leaf_node_id = (0,)  # root node id
        if self.log:
            print('-------- selection ----------')
            print(f"current tree state: \n{self.tree[leaf_node_id]['state']}")

        while not leaf_node_found:
            node_id = leaf_node_id
            n_child = len(self.tree[node_id]['children'])
            if self.log:
                print(f"who's turn is it? {self.tree[node_id]['next_player']}")
                print("checking for children...")
            if n_child == 0:  # no child --> node is a leaf node
                leaf_node_id = node_id
                leaf_node_found = True
            else:
                if self.log:
                    print("tree state has children")
                    print('n_child: ', n_child)

                maximum_ucb_value = -100.0
                for i in range(n_child):  # iterates through all childs (= action possibilities) & finds the action with max ucb value
                    action = self.tree[node_id]['children'][i]
                    child_id = node_id + (action,)  # concatenates node_id and action --> result is e.g. (0, 3)

                    w = self.tree[child_id]['w']
                    n = self.tree[child_id]['n']
                    p = self.tree[child_id]['p']

                    #total_n = self.total_n

                    if n == 0:
                        n = 1e-4  # avoiding division by zero
                    exploitation_value = w / n
                    exploration_value = np.sqrt(self.total_n)/ (1+n)


                    ucb_value = exploitation_value + self.exploration_constant * p * exploration_value
                    if self.log:
                        print(f"child_id: {child_id}, ucb_value: {ucb_value}, n: {n}, w: {w}, p: {p}")

                    if ucb_value > maximum_ucb_value:
                        maximum_ucb_value = ucb_value
                        leaf_node_id = child_id
                if self.log:
                    print(f"total_n: {self.total_n}")
                    print(f"child with max ucb value: {leaf_node_id}")
                    print(f"board of selected child: \n{self.tree[leaf_node_id]['state']}")
                    #print(f"who's turn is it? {self.tree[leaf_node_id]['next_player']}")

        depth = len(leaf_node_id)  # leaf_node_id is a tuple with size = depth
        if depth > self.max_depth_searched:  # added this part
            self.max_depth_searched = depth
        if self.log:
            print('no children, current state is a leaf node')
            print('selected leaf node: ')
            print(self.tree[leaf_node_id])
        return leaf_node_id, depth

    def expansion_evaluation(self, leaf_node_id):
        """ creates all possible outcomes from leaf node and asks the neural evaluation net for probabilities for each child. Stores them in 'p'.
        Moreover, asks the net for a value of the leaf node state (in contrast to pure MCTS, where the value of a random CHILD of the leaf node is
        estimated via a simulation playout with random moves). If state is terminal, value is directly determined by the state. otherwise, value
        is returned by neural network
        in: self.tree, leaf_node_id
        out: expanded tree (self.tree),
        """
        if self.log:
            print('-------- expansion & evaluation----------')
        self.total_n += 1
        leaf_board = deepcopy(self.tree[leaf_node_id]['state'])
        leaf_player = deepcopy(self.tree[leaf_node_id]['next_player'])
        leaf_state = self.create_board_state(leaf_board,leaf_player)

        winner = self._is_terminal(leaf_board)
        possible_actions = self._get_valid_actions(leaf_board) # list of lists, each containing [(row,col), action_idx]
        #action_list = [action_idx for action, action_idx in possible_actions] # extracting the action idx and storing it in a list

        action_probs, value = self.evaluate_state_fn(leaf_state)
        # action_probs: list with probs for all 9 actions (also the impossible ones)

        # updating the child nodes with action_probs from evaluation_net
        if winner is None:  # game is not over yet
            childs = []
            for action_set in possible_actions:
                action, action_idx = action_set  # action is tuple (row,col). action idx always ranges from 0 to 8!
                child_prob = action_probs[action_idx]

                child_state = deepcopy(self.tree[leaf_node_id]['state'])
                next_player = self.tree[leaf_node_id]['next_player']

                if next_player == 'x':
                    child_next_player = 'o'
                    child_state[action] = 1
                else:
                    child_next_player = 'x'
                    child_state[action] = -1

                child_id = leaf_node_id + (action_idx,)
                childs.append(child_id)
                self.tree[child_id] = {'state': child_state,
                                       'next_player': child_next_player,
                                       'children': [],
                                       'parent': leaf_node_id,
                                       'n': 0, 'w': 0, 'q': 0, 'p': child_prob}
                self.tree[leaf_node_id]['children'].append(action_idx)

            if self.log:
                print(f"evaluation of leaf node {leaf_node_id}")
                print(f"it's player {self.tree[leaf_node_id]['next_player']}'s turn now")
                print(f"state being evaluated: \n{self.tree[leaf_node_id]['state']}")
                print(f"value: {value}")

            return value

        else: # game is over, leaf_board is a terminal state
            if self.log:
                print("leaf node is a terminal state")
                print(leaf_board)
                print(f"winner: {winner} (x:1, o:-1)")
            # winner is 'draw', 'x', or 'o'
            if winner == 'draw':
                value = 0  #
            # leaf_player is the player who would play next. value should be positive if the leaf board was caused
            # by the winner, i.e. if the winner is NOT leaf_player
            elif winner != leaf_player:
                value = 1
            else:
                value = -1

            return value


    def _is_terminal(self, leaf_state):
        """checks whether a leaf_state is terminal (i.e. game ends)
        in: game state (np.array)
        out: who wins? ('o', 'x', 'draw', None)
             (None = game not ended)
        """

        def __who_wins(sums, win_mark):
            if np.any(sums == win_mark):
                return 'x'
            if np.any(sums == -win_mark):
                return 'o'
            return None

        def __is_terminal_in_conv(leaf_state, win_mark):
            # check row/col
            for axis in range(2):
                sums = np.sum(leaf_state, axis=axis)
                result = __who_wins(sums, win_mark)
                if result is not None:
                    return result
            # check diagonal
            for order in [-1, 1]:  # checks both diagonals
                diags_sum = np.sum(np.diag(leaf_state[::order]))
                result = __who_wins(diags_sum, win_mark)
                if result is not None:
                    return result
            return None

        # the following part is only important if win_mark and n_rows are not the same (e.g. if board is 5x5 and the winner
        # has to have 3 symbols in a line. Then, 'sliding windows' (i.e. subparts of the board) are created and checked
        # for a winner
        win_mark = self.win_mark  # 3 for tictactoe
        n_rows_board = len(self.tree[(0,)]['state'])
        window_size = win_mark
        window_positions = range(n_rows_board - win_mark + 1)  # range(0,1) for tictactoe

        for row in window_positions:
            for col in window_positions:
                window = leaf_state[row:row + window_size, col:col + window_size]
                winner = __is_terminal_in_conv(window, win_mark)
                if winner is not None:
                    return winner

        # return draw if no more action possible (no empty field on board left)
        if not np.any(leaf_state == 0):
            return 'draw'

        return None

    def _get_valid_actions(self, leaf_state):
        """returns all possible actions in current leaf state
        in:
        - leaf_state (np.array)
        out:
        - set of possible actions (list of lists, each containing [(row,col), action_idx])
            action_idx always ranges from 0 to 8 (including 8)
        """
        actions = []
        count = 0
        state_size = len(leaf_state)

        for i in range(state_size):
            for j in range(state_size):
                if leaf_state[i][j] == 0:
                    actions.append([(i, j), count])
                count += 1
        return actions

    def backprop(self, leaf_node_id, value):
        """travels upwards through the game tree, starting from leaf_node_id and updating the statistics of each
        traversed node
        in: leaf node id, value, self.tree
        out: updated self.tree"""
        orig_player = deepcopy(self.tree[(0,)]['next_player'])
        finish_backprob = False
        node_id = leaf_node_id
        while not finish_backprob:
            if self.log:
                print("tree:")
                print(self.tree)
            self.tree[node_id]['n'] += 1
            next_player = self.tree[node_id]['next_player']
            state = deepcopy(self.tree[node_id]['state'])
#check
            # value here is already aligned to the player of the leaf_node_id. for every subsequent iteration (i.e. traveling
            # one level up the tree), the sign of value has to be changed (as players take turns)
            self.tree[node_id]['w'] += value


            self.tree[node_id]['q'] = self.tree[node_id]['w'] / self.tree[node_id]['n']

            parent_id = self.tree[node_id]['parent']
            if node_id == (0,):
                finish_backprob = True
            if parent_id == (0,):
                parent_state = deepcopy(self.tree[parent_id]['state'])
                self.tree[parent_id]['n'] += 1
                self.tree[parent_id]['w'] += value
                self.tree[parent_id]['q'] = self.tree[parent_id]['w'] / self.tree[parent_id]['n']

                finish_backprob = True

            else:
                node_id = parent_id
                value = -value


#### same function as in ttt.. how to avoid??
    def create_board_state(self, board, next_player):
        """uses information about the next player (i.e. the player whose turn it is) and the board to create a combined
         board state: one array of the current board, and one representing the player (full of -1s or 1s)"""
        board_state = np.zeros((2, board.shape[0], board.shape[1]))
        board_state[0] = board
        # mark represents the next player (all 1s if x is to play, otherwise all -1s)
        if next_player == 'x':
            mark = 1
        elif next_player == 'o':
            mark = -1
        board_state[1][:, :] = mark

        reshaped_board = np.ascontiguousarray(board_state.reshape(-1, 2, board.shape[0], board.shape[1]))

        return reshaped_board


    def solve(self, board, player_symbol, temperature = None, self_play=False):
        """this runs the mcts by calling the selection, expansion, evaluation and backprop functions n_iteration times
        and then selecting the best action. In the end, the tree is updated to keep just the selected best action and its children
        in: board, player_symbol
        out: action, action_probs, updated_tree
        """
        if temperature is None:
            temperature = self.temp

        self._set_tree(board,player_symbol)
        self._set_board(board)
        self._set_player(player_symbol)

        for i in range(self.n_iterations):

            if self.log:
                print(f"\n--------------------iter: {i+1}-----------------------\n")

            leaf_node_id, depth_searched = self.selection()

            if self.log:
                print(f"depth searched: {depth_searched}")

            value = self.expansion_evaluation(leaf_node_id)

            self.backprop(leaf_node_id, value)

            if self.log:
                print(f"leaf node {leaf_node_id} after backpropagation: \n {self.tree[leaf_node_id]}")
                parent_id = self.tree[leaf_node_id]['parent']
                if parent_id:
                    print(f"parent after backpropagation: \n {self.tree[parent_id]}")

            if depth_searched == self.depth:
                if self.log:
                    print("depth_searched == self.depth")
                break

        # SELECT BEST ACTION

        # calc the move probabilities based on visit counts at the root node
        root_id = (0,)
        action_candidates = self.tree[root_id]['children']  # list of children (one number between 0 and 8 per child)
        action_counts = [self.tree[(0,) + (a,)]['n'] for a in action_candidates]
        #print("action_candidates:", action_candidates)
        #print("action_counts:", action_counts)
        all_actions_pi = np.zeros(self.n_rows*self.n_rows) # move probs must be of length 9 for training of network later

        if self_play==True:
            #print("self-play, temp =",temperature)
            action_visits_exp = [(x ** (1. / temperature)) for x in action_counts]
            actions_pi = [x / float(sum(action_visits_exp)) for x in action_visits_exp]
            all_actions_pi[action_candidates] = actions_pi

            best_action = np.random.choice(action_candidates,
                p=0.75 * np.array(actions_pi) + 0.25 * np.random.dirichlet(0.3 * np.ones(len(actions_pi))))
        else:
            # equivalent to chosing action with greatest N
            #print("no self-play, temp =",temperature)
            #print("action_counts",action_counts)
            max_visits = np.argmax(action_counts)
            actions_pi = [0] * len(action_counts)
            actions_pi[max_visits] = 1
            all_actions_pi[action_candidates] = actions_pi
            best_action = np.argmax(all_actions_pi)
        if self.log:
            print("all actions pi:", all_actions_pi)
            print("max action pi idx:",np.argmax(all_actions_pi))
            print("chosen action idx:",best_action)

        best_n = -100
        best_q = -100
        for a in action_candidates:
            n = self.tree[(0,) + (a,)]['n']
            q = self.tree[(0,) + (a,)]['q']
            if n > best_n:
                best_n = n
                best_action_n = a
            if q > best_q:
                best_q = q
        #print(best_action == best_action_n)

        if self.log:
            # FOR DEBUGGING
            print('\n-----------Summary of AlphaMCTS Results-----------')
            print(' [-] game board: \n')
            for row in self.tree[(0,)]['state']:
                print(row)
            print(' [-] person to play: ', self.tree[(0,)]['next_player'])
            print('\n [-] best_action: %d' % best_action)
            print(' best_q = %.2f' % (best_q))
            print(' best_n = %.2f' % (best_n))
            print(' [-] searching depth = %d' % (depth_searched))
            print(f"[-] max_depth_searched: {self.max_depth_searched}")
        if self.fig:
            # FOR DEBUGGING
            fig = plt.figure(figsize=(5, 6))
            for a in action_candidates:
                # print('a= ', a)
                _node = self.tree[(0,) + (a,)]
                _state = deepcopy(_node['state'])
                _q = _node['q']
                _n = _node['n']
                # print(_state)
                plt.subplot(len(_state), len(_state), a + 1)
                plt.pcolormesh(_state, alpha=0.7, cmap="RdBu")
                # plt.axis('equal')
                plt.gca().invert_yaxis()
                plt.xticks([], [])
                plt.yticks([], [])
                plt.title('[%d] q=%.2f, \nn=%.2f' % (a, _q, _n))
            plt.tight_layout()
            plt.draw()
            plt.waitforbuttonpress(0)
            plt.close(fig)

        if self.log:
            print("new tree")
            print(self.tree)

        return best_action, best_n, best_q, depth_searched, all_actions_pi, self.tree


'''
for test
'''
if __name__ == '__main__':
    tic = TTT()
    # tic.make_move((0, 0), -1)
    # tic.make_move((1, 1),  1)
    # tic.make_move((2, 0), -1)
    # tic.make_move((2, 2),  1)
    tic.board = np.array(([-1, 0, 0],
                          [0, 1, 0],
                          [-1, 0, 1]))
    # tic.board = np.array(([-1, 1, 0],
    #                       [0, 1, 1],
    #                      [-1, -1, 1]))
    mcts = AlphaZeroMCTS(n_iterations=500, depth=10, exploration_constant=1.4,
                       game_board=tic.board, tree=None, win_mark=3, player='x', log=True, fig=True)
    best_action, best_n, best_q, depth_searched,move_probs,tree = mcts.solve()
    print('best action= ', best_action, ' best_n= ', best_n, ' best_q= ', best_q, 'depth_searched=', depth_searched,
          'move_probs=',move_probs,'tree=',tree)



