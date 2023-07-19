#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway
from vocab import VocabEntry
import torch
# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab:VocabEntry):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        pad_token_idx = vocab.char2id['<pad>']
        self.embed_size = embed_size
        char_embed_size = 50
        self.char_embedding = nn.Embedding(len(vocab.char2id),
                                           char_embed_size,
                                           pad_token_idx)
        self.convNN = CNN(f=self.embed_size)
        self.highway = Highway(embed_size=embed_size)
        self.dropout = nn.Dropout(p=0.3)

        ### END YOUR CODE

    def forward(self, input:torch.Tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code
        # print(input.shape) #10,5,21
        ### YOUR CODE HERE for part 1j
        X_word_emb_list = []
        # divide input into sentence_length batchs
        for X_padded in input:
            # print(X_padded.shape) #5,21
            X_emb = self.char_embedding(X_padded)
            # print(X_emb.shape) #5,21,50
            X_reshaped = torch.transpose(X_emb, dim0=-1, dim1=-2)
            # print(X_reshaped.shape) # 5,50,21
            # conv1d can only take 3-dim mat as input
            # so it needs to concat/stack all the embeddings of word
            # after going through the network
            X_conv_out = self.convNN(X_reshaped)
            # print(X_conv_out.shape) #5,3
            X_highway = self.highway(X_conv_out)
            # print(X_highway.shape) #5,3
            X_word_emb = self.dropout(X_highway)
            # print(X_word_emb.shape) #5,3
            X_word_emb_list.append(X_word_emb)

        X_word_emb = torch.stack(X_word_emb_list)
        # print(X_word_emb.shape) #10,5,3
        return X_word_emb


        ### END YOUR CODE

