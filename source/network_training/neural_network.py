#from source.game import tictactoe
#from source.network_training import mcts
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import argparse
import os
import shutil
import time
import random
import math
import sys
sys.path.append('../../')
from utils import *

class NeuralNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.board_x, self.board_y = 3, 3 # board size
        self.n_actions = 9 # number of possible actions
        self.dropout = 0.2 # proportion of dropout neurons

        self.num_filters = 6 # change this later?? = nr of convolutional filters  (kernels) applied

        # common layers for probs and value
        self.conv1 = nn.Conv2d(2, self.num_filters, kernel_size = 3, stride=1, padding=1) # nr of output channels = nr of filters (kernels) used in the conv layer
        self.conv2 = nn.Conv2d(self.num_filters, self.num_filters, kernel_size = 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.num_filters, self.num_filters*2, kernel_size = 3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_filters)
        self.bn2 = nn.BatchNorm2d(self.num_filters)
        self.bn3 = nn.BatchNorm2d(self.num_filters*2)

        # probs layers
        self.probs_conv = nn.Conv2d(self.num_filters*2, 2, kernel_size = 1, stride=1)
        self.bn_probs_conv = nn.BatchNorm2d(2)
        self.probs_fc1 = nn.Linear(2 * self.board_x * self.board_y, self.board_x * self.board_y)     # output should be 9

        # value layers
        self.val_conv = nn.Conv2d(self.num_filters*2, 2, kernel_size = 1)
        self.bn_val_conv = nn.BatchNorm2d(2)
        self.val_fc1 = nn.Linear(2*self.board_x*self.board_y,12)
        self.val_fc2 = nn.Linear(12,1)

    def forward(self, state):

        # all input goes through the 3 conv layers
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # probs layers
        x_probs = F.relu(self.bn_probs_conv(self.probs_conv(x)))
        x_probs = x_probs.view(-1, 2* self.board_x*self.board_y)
        x_probs = F.log_softmax(self.probs_fc1(x_probs),dim = 1) # dim = dimension along which softmax will be computed

        # value layers
        x_val = F.dropout(F.relu(self.bn_val_conv(self.val_conv(x))), p = self.dropout, training = self.training)
        x_val = x_val.view(-1, 2 * self.board_x * self.board_y)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))

        return x_probs, x_val


class EvaluationNet():

    def __init__(self, model_file=None):

        self.board_width = 3
        self.board_height = 3
        self.l2_const = 1e-4  # coef of l2 penalty

        self.net_module = NeuralNet()
        self.optimizer = optim.Adam(self.net_module.parameters(), weight_decay=self.l2_const)

        if model_file:
            #https://pytorch.org/tutorials/beginner/saving_loading_models.html
            print("EvalNet loading model_file")
            net_params = torch.load(model_file)
            self.net_module.load_state_dict(net_params)


    def evaluate_batch(self, states_batch):
        """
        in: a batch of states (each consists of board + info about next_player)
        out: a batch of action probabilities and state values
        """
        states_batch = Variable(torch.FloatTensor(states_batch))
        log_act_probs, value = self.net_module(states_batch)
        action_probs = np.exp(log_act_probs.data.numpy())

        return action_probs, value.data.numpy()

    def evaluate_state_fn(self, state):
        """
        in: a state (consists of board + info about next_player
        out: list of nine probabilities (for each action, even the impossible ones!) and the value of the board state
                e.g. [0.01, 0.2,....,0.3], -1
        """
        log_act_probs, value = self.net_module(Variable(torch.from_numpy(state)).float())
        act_probs = np.exp(log_act_probs.data.numpy().flatten())
        value = value.data.item()

        return act_probs, value

    def train_step(self, states_batch, mcts_probs_batch, winners_batch, lr):
        """conduct one step of training of the evaluation net"""
        self.net_module.train()
        states_batch = Variable(torch.FloatTensor(states_batch))
        mcts_probs = Variable(torch.FloatTensor(mcts_probs_batch))
        winners_batch = Variable(torch.FloatTensor(winners_batch))

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # set learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # forward
        log_act_probs, value = self.net_module(states_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2

# check loss functions
        value_loss = F.mse_loss(value.view(-1), winners_batch)
        probs_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        loss = value_loss + probs_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
            torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
        )
        return loss.data.item(), entropy.data.item()


    def save_model(self, file_path):
        """ save model params to file """
        net_params = self.net_module.state_dict()  # get model params
        torch.save(net_params, file_path)


    '''
    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict' : self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        checkpoint = torch.load(filepath)
        self.nnet.load_state_dict(checkpoint['state_dict'])
    '''


