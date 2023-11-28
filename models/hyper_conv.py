import datetime
import math
import numpy as np
import torch
from torch import nn, backends
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse
from scipy.sparse import coo
import time
import heapq

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


class HyperConv(Module):
    def __init__(self, layers,dataset,emb_size=100):
        super(HyperConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.dataset = dataset

    def forward(self, adjacency, embedding):
        item_embeddings = embedding
        item_embedding_layer0 = item_embeddings
        final = [item_embedding_layer0]
        for i in range(self.layers):
            item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), item_embeddings)
            final.append(item_embeddings)
      #  final1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in final]))
      #  item_embeddings = torch.sum(final1, 0)
        item_embeddings = np.sum(final, 0) / (self.layers+1)
        return item_embeddings