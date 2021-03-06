import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from source.game.tictactoe import TTT

class PureMCTS(object):
    def __init__(self, n_iterations=50, depth=15, exploration_constant=5.0, tree = None, win_mark=3, game_board=None, player=None, log = False, fig = True):
        """@tree: nested dictionary, format see _set_tictactoe
        @ win_mark: if the row-wise, col-wise or diagonal-wise sum reaches this number, player 'x' (represented by 1 on the board)
            has won. If the sum = -win_mark, the player 'o' (using -1) has won
        @ player: player whose turn it is now. 'o' or 'x'
        @ log: boolean, if True infos will be printed to console
        @ fig: boolean, if True figure with stats will be shown"""

        self.n_iterations = n_iterations
        self.depth = depth
        self.exploration_constant = exploration_constant
        self.total_n = 0 #1e-4 # orig 0
        self.max_depth_searched = 0
        self.game_board = game_board

        self.log = log
        self.fig = fig

        self.orig_player = player
        self.root_id = (0,)
        self.leaf_node_id = None
        self.n_rows = len(game_board) # normal tictactoe: 3
        self.win_mark = win_mark # later used to check victory (tictactoe: 3 in a row)

        if tree == None:
            self._set_tree(game_board, player)
        else:
            self.tree = tree
            if self.log:
                print("TREE IS REUSED")

    def _set_player(self,player):
        self.orig_player = player
        self.tree[self.root_id]['next_player'] = self.orig_player

    def _set_board(self,board):
        self.game_board = board


    def _set_tree(self, game_board, player):
        """creates a nested tree dictionary with key = node_id, value = dict with keys state, player, child, parent, n, w, q.
        n, w and q will be filled later with: n = visit count of node, w = win count, q = w/n"""
        #self.root_id = (0,)
        self.tree = {self.root_id: {'state': game_board,
                          'next_player': player,
                          'child': [],
                          'parent': None,
                          'n': 0,
                          'w': 0,
                          'q': None}}

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
        - depth (depth of node root=0)
        """
        leaf_node_found = False
        leaf_node_id = (0,) # root node id
        if self.log:
            print('-------- selection ----------')
            print("NEXT PLAYER:",self.tree[leaf_node_id]['next_player'])

        while not leaf_node_found:
            node_id = leaf_node_id
            n_child = len(self.tree[node_id]['child'])
            if self.log:
                print(f"current tree state: \n{self.tree[node_id]['state']}")
                print(f"next player: {self.tree[node_id]['next_player']}")
                print("checking for children...")
            if n_child == 0: # no child --> node is a leaf node
                leaf_node_id = node_id
                leaf_node_found = True
            else:
                if self.log:
                    print("tree state has children")
                    print('n_child: ', n_child)

                maximum_ucb_value = -100.0
                for i in range(n_child): # iterates through all childs (= action possibilities) & finds the action with max ucb value
                    action = self.tree[node_id]['child'][i]
                    child_id = node_id + (action,) # concatenates node_id and action --> result is e.g. (0, 3)

                    w = self.tree[child_id]['w']
                    n = self.tree[child_id]['n']
                    total_n = self.total_n
                    parent_id = self.tree[node_id]['parent']

                    '''
                    if parent_id == None:
                        total_n = 1
                    else:
                        total_n = self.tree[parent_id]['n']
                    '''

                    if n == 0:
                        n = 1e-4 # avoiding division by zero
                    exploitation_value = w / n
                    exploration_value  = np.sqrt(np.log(total_n)/n)
                    ucb_value = exploitation_value + self.exploration_constant * exploration_value
                    if self.log:
                        print(f"child_id: {child_id}, ucb_value: {ucb_value}, n: {n}, w: {w}")


                    if ucb_value > maximum_ucb_value:
                        maximum_ucb_value = ucb_value
                        leaf_node_id = child_id
                if self.log:
                    print(f"total_n: {total_n}")
                    print(f"child with max ucb value: {leaf_node_id}")
                    #print(f"who's turn is it? {self.tree[leaf_node_id]['next_player']}")


        depth = len(leaf_node_id) # leaf_node_id is a tuple with size = depth
        if depth > self.max_depth_searched: # added this part
            self.max_depth_searched = depth
        if self.log:
            print('no children, current state is a leaf node')
            print('selected leaf node: ')
            print(self.tree[leaf_node_id])
        return leaf_node_id, depth

    def expansion(self, leaf_node_id):
        """ creates all possible outcomes from leaf node and randomly selects one
        in: self.tree, leaf_node_id
        out: expanded tree (self.tree),
             randomly selected child node id (child_node_id)
        """
        if self.log:
            print('-------- expansion ----------')
        leaf_state = self.tree[leaf_node_id]['state']
        winner = self._is_terminal(leaf_state)
        possible_actions = self._get_valid_actions(leaf_state)

        child_node_id = leaf_node_id # default value, if leaf node is terminal state this is returned
        if winner is None: # when leaf node is not a terminal state
            childs = []
            for action_set in possible_actions:
                action, action_idx = action_set # action is tuple (row,col). action idx always ranges from 0 to 8!
                state = deepcopy(self.tree[leaf_node_id]['state'])
                next_player = self.tree[leaf_node_id]['next_player']

                if next_player == 'x':
                    child_next_player = 'o'
                    state[action] = 1
                else:
                    child_next_player = 'x'
                    state[action] = -1

                child_id = leaf_node_id + (action_idx, )
                childs.append(child_id)
                self.tree[child_id] = {'state': state,
                                       'next_player': child_next_player,
                                       'child': [],
                                       'parent': leaf_node_id,
                                       'n': 0, 'w': 0, 'q':0}
                self.tree[leaf_node_id]['child'].append(action_idx)
            rand_idx = np.random.randint(low=0, high=len(childs))
            if self.log:
                print('tree was expanded:')
                print('childs: ', childs)
                print(f"rand_idx: {rand_idx}")
            child_node_id = childs[rand_idx]
            if self.log:
                print(f"randomly selected child id: {child_node_id}")
        else:
            if self.log:
                print("leaf node is a terminal state")
        return child_node_id

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
            for order in [-1,1]: # checks both diagonals
                diags_sum = np.sum(np.diag(leaf_state[::order]))
                result = __who_wins(diags_sum, win_mark)
                if result is not None:
                    return result
            return None

        # the following part is only important if win_mark and n_rows are not the same (e.g. if board is 5x5 and the winner
        # has to have 3 symbols in a line. Then, 'sliding windows' (i.e. subparts of the board) are created and checked
        # for a winner
        win_mark = self.win_mark # 3 for tictactoe
        n_rows_board = len(self.tree[(0,)]['state'])
        window_size = win_mark
        window_positions = range(n_rows_board - win_mark + 1) # range(0,1) for tictactoe

        for row in window_positions:
            for col in window_positions:
                window = leaf_state[row:row+window_size, col:col+window_size]
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

    def simulation(self, child_node_id):
        """simulate game from child node's state until it reaches the terminal state of the game.
        in:
        - child node id (randomly selected child node id from `expansion`)
        out:
        - winner ('o', 'x', 'draw')"""
        if self.log:
            print('-------- simulation ----------')
        self.total_n += 1
        state = deepcopy(self.tree[child_node_id]['state'])
        child_player = deepcopy(self.tree[child_node_id]['next_player'])
        anybody_win = False
        if self.log:
            print(f"simulation starting from child node {child_node_id}")
            print(f"it's player {self.tree[child_node_id]['next_player']}'s turn now")
            print(f"simulation starting state: \n{self.tree[child_node_id]['state']}")

        while not anybody_win:
            winner = self._is_terminal(state)
            if winner is not None: # we have a winner!
                if self.log:
                    print('end of simulation state')
                    print(state)
                    print(f"winner: {winner} (x:1, o:-1)")
                anybody_win = True
            else:
                possible_actions = self._get_valid_actions(state) #
                # randomly choose action for simulation (= random rollout policy)
                rand_idx = np.random.randint(low=0, high=len(possible_actions))
                action, _ = possible_actions[rand_idx]

                if child_player == 'o': # player is always the one who will make the next move from the current board
                    player_next_turn = 'x'
                    state[action] = -1
                else:
                    player_next_turn = 'o'
                    state[action] = 1

                child_player = player_next_turn
        return winner

    def backprop(self, child_node_id, winner):
        """travels upwards through the game tree, starting from child_node_id and updating the statistics of each
        traversed node
        in: child node id, winner ('draw', 'x', or 'o'), self.tree
        out: updated self.tree"""
#        orig_player = deepcopy(self.tree[(0,)]['next_player'])
#
        child_next_player = deepcopy(self.tree[child_node_id]['next_player'])
        #print('orig root:',deepcopy(self.tree[(0,)]['next_player']))
        #print('orig child:',deepcopy(self.tree[child_node_id]['next_player']))
        # winner is 'x', 'o', or 'draw'
        if winner == 'draw':
            reward = 0
        # if the winner is the player who chose child_node_id (i.e. who is NOT next_player of child_id), reward is +1
        elif winner != child_next_player:
            reward = 1
        else:
            reward = -1

        finish_backprob = False
        node_id = child_node_id
        while not finish_backprob:
            self.tree[node_id]['n'] += 1

            # all traversed nodes that were caused by the same player as child_node will be rewarded +1
            if self.tree[node_id]['next_player'] == child_next_player:
                self.tree[node_id]['w'] += reward
            elif self.tree[node_id]['next_player'] != child_next_player:
                self.tree[node_id]['w'] += -reward

            ## original: self.tree[node_id]['w'] += reward
            self.tree[node_id]['q'] = self.tree[node_id]['w'] / self.tree[node_id]['n']
            parent_id = self.tree[node_id]['parent']
            if parent_id == (0,):
                self.tree[parent_id]['n'] += 1

                if self.tree[parent_id]['next_player'] != child_next_player:
                    self.tree[parent_id]['w'] += reward
                elif self.tree[parent_id]['next_player'] == child_next_player:
                    self.tree[parent_id]['w'] += -reward

                #self.tree[parent_id]['w'] += 0#reward
                self.tree[parent_id]['q'] = self.tree[parent_id]['w'] / self.tree[parent_id]['n']
                finish_backprob = True
            else:
                node_id = parent_id

    def solve(self, board, player_symbol):
        """this runs the mcts by calling the selection, expansion, simulation and backprop functions n_iteration times
        and then selecting the best action (highest n value)"""

        self._set_tree(board,player_symbol)
        self._set_board(board)
        self._set_player(player_symbol)
        for i in range(self.n_iterations):

            if self.log:
                print(f"\n--------------------iter: {i+1}-----------------------\n")

            leaf_node_id, depth_searched = self.selection()

            if self.log:
                print(f"depth searched: {depth_searched}")

            child_node_id = self.expansion(leaf_node_id)

            winner = self.simulation(child_node_id)
            self.backprop(child_node_id, winner)


            if depth_searched == self.depth:
                if self.log:
                    print("depth_searched == self.depth")
                break
        # SELECT BEST ACTION
        current_state_node_id = (0,)
        action_candidates = self.tree[current_state_node_id]['child'] # list of children (one number between 0 and 8 per child)
        if self.log:
            print("ACTION CAND:",action_candidates)
        #print("node next player",self.tree[current_state_node_id]['next_player'])
        # qs = [self.tree[(0,)+(a,)]['q'] for a in action_candidates]
        best_n = -100
        best_q = -100
        for a in action_candidates:
            n = self.tree[(0,) + (a,)]['n']
            q = self.tree[(0,) + (a,)]['q']
            if n > best_n:
                best_n = n
                best_action = a
            if q > best_q:
                best_q = q


        """
        in original vanilla mcts:
        
        best_q = -100
        for a in action_candidates:
            q = self.tree[(0,)+(a,)]['q']
            if q > best_q:
                best_q = q
                best_action = a # used in orig
        """

        if self.log:
            # FOR DEBUGGING
            print('\n-----------Summary of PureMCTS Results-----------')
            print(' [-] game board: \n')
            for row in self.tree[(0,)]['state']:
                print (row)
            print(' [-] person to play: ', self.tree[(0,)]['next_player'])
            print('\n [-] best_action: %d' % best_action)
            #print(' best_q = %.2f' % (best_q))
            print(' best_n = %.2f' % (best_n))
            print(' [-] searching depth = %d' % (depth_searched))
            print(f"[-] max_depth_searched: {self.max_depth_searched}\n\n\n\n")
        if self.fig:
            # FOR DEBUGGING
            fig = plt.figure(figsize=(5,6))
            for a in action_candidates:
                #print('a= ', a)
                _node = self.tree[(0,)+(a,)]
                _state = deepcopy(_node['state'])
                _q = _node['q']
                _n = _node['n']
                #print(_state)
                plt.subplot(len(_state),len(_state),a+1)
                plt.pcolormesh(_state, alpha=0.7, cmap="RdBu")
                #plt.axis('equal')
                plt.gca().invert_yaxis()
                plt.xticks([], [])
                plt.yticks([], [])
                plt.title('[%d] q=%.2f, \nn=%.2f' % (a,_q,_n))
            plt.tight_layout()
            plt.draw()
            plt.waitforbuttonpress(0)
            plt.close(fig)


        if self.log:
            print("tree: ")
            print(self.tree)


        return best_action, best_n, best_q, depth_searched


'''
for test
'''
if __name__ == '__main__':

    tic = TTT()
    #tic.make_move((0, 0), -1)
    #tic.make_move((1, 1),  1)
    #tic.make_move((2, 0), -1)
    #tic.make_move((2, 2),  1)
    tic.board = np.array(([-1, 0, 0],
                           [0, 1, 0],
                          [-1, 0, 1]))
    #tic.board = np.array(([-1, 1, 0],
    #                       [0, 1, 1],
    #                      [-1, -1, 1]))
    mcts = PureMCTS(n_iterations=10, depth=10, exploration_constant=1.4,
                    game_board = tic.board, tree = None, win_mark=3, player='x', log=True, fig = False)
    best_action, best_n, best_q, depth_searched= mcts.solve()
    print('best action= ', best_action, ' best_n= ', best_n,' best_q= ', best_q, 'depth_searched=', depth_searched)

    """
    tic.board = np.array(([-1, 0, 0],
                           [1, 1, 0],
                          [-1, 0, 1]))
    mcts2 = PureMCTS(n_iterations=100, depth=10, exploration_constant=1.4,
                    game_board=tic.board, tree=None, win_mark=3, player='o', log=True, fig=False)
    best_action, best_n, best_q, depth_searched, = mcts2.solve()
    print('best action= ', best_action, ' best_n= ', best_n,' best_q= ', best_q, 'depth_searched=', depth_searched)
    """



    # testing 1 iteration
    # leaf_node_id, depth = mcts.selection()
    # child_node_id = mcts.expansion(leaf_node_id)
    #
    # print('child node id = ', child_node_id)
    # print(' [*] simulation ...')
    # winner = mcts.simulation(child_node_id)
    # print(' winner', winner)
    # mcts.backprop(child_node_id, winner)

