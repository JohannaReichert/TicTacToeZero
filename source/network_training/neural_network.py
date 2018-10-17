from source.game import tictactoe
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
from pytorch_classification.utils import Bar, AverageMeter

class NeuralNet(nn.Module):

    def __init__(self,game,args= {}):
        super().__init__()
        self.board_x, self.board_y = 3, 3 # board size
        self.n_actions = 9 # number of possible actions
        self.args = args
        self.dropout = 0.2 # proportion of dropout neurons
        self.batch_size = 64

        self.num_filters = 32 # change this later?? = nr of convolutional filters  (kernels) applied, must be a power of 2 between 32 and 1024

        self.conv1 = nn.Conv2d(1, self.num_filters, 2, stride=1, padding=1) # nr of output channels = nr of filters (kernels) used in the conv layer
        self.conv2 = nn.Conv2d(self.num_filters, self.num_filters, 2, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.num_filters, self.num_filters*2, 2, stride=1)
        self.conv4 = nn.Conv2d(self.num_filters*2, self.num_filters*4, 2, stride=1)

        self.bn1 = nn.BatchNorm2d(self.num_filters)
        self.bn2 = nn.BatchNorm2d(self.num_filters)
        self.bn3 = nn.BatchNorm2d(self.num_filters*2)
        self.bn4 = nn.BatchNorm2d(self.num_filters*2)

        self.fc1 = nn.Linear(self.num_filters*2, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.n_actions)

        self.fc4 = nn.Linear(512, 1)


    def forward(self, state):

        state = state.view(-1, 1, self.board_x, self.board_y)
        state = F.relu(self.bn1(self.conv1(state)))
        state = F.relu(self.bn2(self.conv2(state)))
        state = F.relu(self.bn3(self.conv3(state)))
        state = F.relu(self.bn4(self.conv4(state)))
        state = state.view(-1, self.num_filters*2)

        state = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.dropout, training=self.training)  # batch_size x 1024
        state = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.dropout, training=self.training)  # batch_size x 512

        probs = self.fc3(state)                                                                         # batch_size x action_size
        value = self.fc4(state)                                                                          # batch_size x 1

        return F.log_softmax(probs, dim=1), F.tanh(value)


    def training(self,instances):

        """
           instances: list, each item of form (board, probs, value)
        """
        optimizer = optim.Adam(self.parameters())

        for epoch in range(10):
            print('EPOCH ::: ' + str(epoch + 1))
            self.train()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            probs_losses = AverageMeter()
            value_losses = AverageMeter()
            end = time.time()

            bar = Bar('Training Net', max=int(len(instances) / self.batch_size))
            batch_idx = 0

            while batch_idx < int(len(instances) / args.batch_size):
                sample_ids = np.random.randint(len(instances), size=self.batch_size)
                boards, probs, values = list(zip(*[instances[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_probs = torch.FloatTensor(np.array(probs))
                target_valuess = torch.FloatTensor(np.array(values).astype(np.float64))

                boards, target_probs, target_values = Variable(boards), Variable(target_probs), Variable(target_values)

                # measure data loading time
                data_time.update(time.time() - end)

                # compute output
                out_prob, out_value = self.nnet(boards)
                l_pi = self.loss_pi(target_probs, out_prob)
                l_v = self.loss_v(target_values, out_value)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.data[0], boards.size(0))
                v_losses.update(l_v.data[0], boards.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                batch_idx += 1

                # plot progress
                bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}'.format(
                    batch=batch_idx,
                    size=int(len(examples) / args.batch_size),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    lpi=pi_losses.avg,
                    lv=v_losses.avg,
                )
                bar.next()
            bar.finish()





    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if args.cuda: board = board.contiguous().cuda()
        board = Variable(board, volatile=True)
        board = board.view(1, self.board_x, self.board_y)

        self.nnet.eval()
        pi, v = self.nnet(board)

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets*outputs)/targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]

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



net = NeuralNet()
print(net)


"""


########################################################################
# You just have to define the ``forward`` function, and the ``backward``
# function (where gradients are computed) is automatically defined for you
# using ``autograd``.
# You can use any of the Tensor operations in the ``forward`` function.
#
# The learnable parameters of a model are returned by ``net.parameters()``

params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

########################################################################
# Let try a random 32x32 input
# Note: Expected input size to this net(LeNet) is 32x32. To use this net on
# MNIST dataset, please resize the images from the dataset to 32x32.

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

########################################################################
# Zero the gradient buffers of all parameters and backprops with random
# gradients:
net.zero_grad()
out.backward(torch.randn(1, 10))

########################################################################
# .. note::
#
#     ``torch.nn`` only supports mini-batches. The entire ``torch.nn``
#     package only supports inputs that are a mini-batch of samples, and not
#     a single sample.
#
#     For example, ``nn.Conv2d`` will take in a 4D Tensor of
#     ``nSamples x nChannels x Height x Width``.
#
#     If you have a single sample, just use ``input.unsqueeze(0)`` to add
#     a fake batch dimension.
#
# Before proceeding further, let's recap all the classes you’ve seen so far.
#
# **Recap:**
#   -  ``torch.Tensor`` - A *multi-dimensional array* with support for autograd
#      operations like ``backward()``. Also *holds the gradient* w.r.t. the
#      tensor.
#   -  ``nn.Module`` - Neural network module. *Convenient way of
#      encapsulating parameters*, with helpers for moving them to GPU,
#      exporting, loading, etc.
#   -  ``nn.Parameter`` - A kind of Tensor, that is *automatically
#      registered as a parameter when assigned as an attribute to a*
#      ``Module``.
#   -  ``autograd.Function`` - Implements *forward and backward definitions
#      of an autograd operation*. Every ``Tensor`` operation, creates at
#      least a single ``Function`` node, that connects to functions that
#      created a ``Tensor`` and *encodes its history*.
#
# **At this point, we covered:**
#   -  Defining a neural network
#   -  Processing inputs and calling backward
#
# **Still Left:**
#   -  Computing the loss
#   -  Updating the weights of the network
#
# Loss Function
# -------------
# A loss function takes the (output, target) pair of inputs, and computes a
# value that estimates how far away the output is from the target.
#
# There are several different
# `loss functions <http://pytorch.org/docs/nn.html#loss-functions>`_ under the
# nn package .
# A simple loss is: ``nn.MSELoss`` which computes the mean-squared error
# between the input and the target.
#
# For example:

output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

########################################################################
# Now, if you follow ``loss`` in the backward direction, using its
# ``.grad_fn`` attribute, you will see a graph of computations that looks
# like this:
#
# ::
#
#     input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
#           -> view -> linear -> relu -> linear -> relu -> linear
#           -> MSELoss
#           -> loss
#
# So, when we call ``loss.backward()``, the whole graph is differentiated
# w.r.t. the loss, and all Tensors in the graph that has ``requires_grad=True``
# will have their ``.grad`` Tensor accumulated with the gradient.
#
# For illustration, let us follow a few steps backward:

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

########################################################################
# Backprop
# --------
# To backpropagate the error all we have to do is to ``loss.backward()``.
# You need to clear the existing gradients though, else gradients will be
# accumulated to existing gradients.
#
#
# Now we shall call ``loss.backward()``, and have a look at conv1's bias
# gradients before and after the backward.


net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

########################################################################
# Now, we have seen how to use loss functions.
#
# **Read Later:**
#
#   The neural network package contains various modules and loss functions
#   that form the building blocks of deep neural networks. A full list with
#   documentation is `here <http://pytorch.org/docs/nn>`_.
#
# **The only thing left to learn is:**
#
#   - Updating the weights of the network
#
# Update the weights
# ------------------
# The simplest update rule used in practice is the Stochastic Gradient
# Descent (SGD):
#
#      ``weight = weight - learning_rate * gradient``
#
# We can implement this using simple python code:
#
# .. code:: python
#
#     learning_rate = 0.01
#     for f in net.parameters():
#         f.data.sub_(f.grad.data * learning_rate)
#
# However, as you use neural networks, you want to use various different
# update rules such as SGD, Nesterov-SGD, Adam, RMSProp, etc.
# To enable this, we built a small package: ``torch.optim`` that
# implements all these methods. Using it is very simple:

import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update


###############################################################
# .. Note::
#
#       Observe how gradient buffers had to be manually set to zero using
#       ``optimizer.zero_grad()``. This is because gradients are accumulated
#       as explained in `Backprop`_ section.


"""

'''


class NeuralNet():

    dtype = torch.float
    device = torch.device("cpu")

    # use working example instead
    inputs =
    target1 =
    target2 =

    N = inputs.shape[0]
    D_in = inputs.shape[1]
    D_out = targets.max().values[0] + 1
    H = 9

    print(N, D_in, H, D_out)

    x = torch.tensor(inputs.values, device=device, dtype=dtype)
    y = torch.tensor(targets.values, device=device, dtype=torch.long).squeeze()

    # Hyper-parameters
    learning_rate = 0.0005
    batch_size = 64

    # Neuronal Network
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, D_out)
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_hist = []

    # Train
    epochs = range(2000)
    idx = 0
    for t in epochs:
        for batch in range(0, int(N / batch_size)):
            # Berechne den Batch

            batch_x = x[batch * batch_size: (batch + 1) * batch_size, :]
            batch_y = y[batch * batch_size: (batch + 1) * batch_size]

            # Berechne die Vorhersage (foward step)
            outputs1, outputs2 = model(batch_)


            # Berechne den Fehler (Ausgabe des Fehlers alle 100 Iterationen)
            loss1 = criterion1(outputs1, target1)
            loss2 = criterion2(outputs2, target2)
            loss = loss1 + loss2

            # Berechne die Gradienten und Aktualisiere die Gewichte (backward step)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Berechne den Fehler (Ausgabe des Fehlers alle 100 Iterationen)
        if t % 50 == 0:
            loss_hist.append(loss.item())
            print(t, loss.item())
            torch.save(model, 'vanilla.pt')

    torch.save(model, 'vanilla.pt')

    def __init__(self,game):
        self.game = game
        pass


    def predict_probs(self,board):
        """for now, just return random vals
        outputs action probabilities for every action on the board (len(probs) == 9!!)

        """

        action_len = 9
        return np.random.rand(action_len)


    def predict_value(self,board,next_player):
        """for now, just return a random val
        outputs value of the state for the current player"""
        return np.random.rand(1)*10

    def train(self):
        pass
'''
