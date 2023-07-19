#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
### YOUR CODE HERE for part 1h

class Highway(nn.Module):
    """
    Simple highway neural network
    """
    def __init__(self,embed_size, activation=nn.functional.relu) -> None:
        """ Init Highway Model.

        @param embedd_size (int): input size (dimensionality)
        @param activation (nn.functional): relu function
        """
        super(Highway, self).__init__()
        
        self.activation = activation
        self.W_proj = nn.Linear(embed_size , embed_size,bias=True)
        self.W_gate = nn.Linear(embed_size,embed_size,bias=True)
    def forward(self,x_conv_out:torch.Tensor) -> torch.Tensor: # -> x_highway
        """ Take x_conv_out and convert into x_highway

        @param x_conv_out (Tensor): Output from ConvNet

        @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each example in the input batch. Here b = batch size.
        """
        t = torch.sigmoid(self.W_proj(x_conv_out))
        h = self.activation(self.W_gate(x_conv_out))
        y = t * h + (1 - t) * x_conv_out
        return y
### END YOUR CODE 
