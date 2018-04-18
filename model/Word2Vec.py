from torch.autograd import Variable
import torch.nn as nn
import torch
import numpy as np

class Bundler(nn.Module):
    def forward(self, data):
        raise NotImplementedError

    def forward_i(self, data):
        raise NotImplementedError

    def forward_o(self, data):
        raise NotImplementedError

class Word2Vec(Bundler):
    def __init__(self, vocab_size, embedding_size=300, padding_idx=0):
        super(Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.ivectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ovectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ivectors.weight = nn.Parameter(torch.cat([torch.zeros(1, self.embedding_size), torch.FloatTensor(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        self.ovectors.weight = nn.Parameter(torch.cat([torch.zeros(1, self.embedding_size), torch.FloatTensor(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        self.ivectors.weight.requires_grad = True
        self.ovectors.weight.requires_grad = True

    def forward(self, data):
        return self.forward_i(data)

    def forward_i(self, data):
        v = Variable(data, requires_grad=False)
        v = v.cuda() if self.ivectors.weight.is_cuda else v
        return self.ivectors(v)

    def forward_o(self, data):
        v = Variable(data, requires_grad=False)
        v = v.cuda() if self.ivectors.weight.is_cuda else v
        return self.ovectors(v)

class SkipGram(nn.Module):
    def __init__(self, embedding, vocab_size, n_negs=20, weights=None):
        super(SkipGram, self).__init__()
        self.embedding = embedding
        self.vocab_size = vocab_size
        self.n_negs = n_negs
        self.weights = None
        if weights is not None:
            wf = np.power(weights, 0.75)
            wf = wf / wf.sum()
            self.weights = torch.from_numpy(wf)

    def forward(self, iwords, owords):
        batch_size = iwords.size()[0]
        context_size = owords.size()[1]
        if self.weights is not None:
            nwords = torch.multinomial(self.weights, batch_size * context_size * self.n_negs, replacement=True).view(batch_size, -1)
        else:
            nwords = torch.FloatTensor(batch_size, context_size * self.n_negs).uniform_(0, self.vocab_size - 1).long()
        
        ivectors = self.embedding.forward_i(iwords).unsqueeze(2)
        ovectors = self.embedding.forward_o(owords)
        nvectors = self.embedding.forward_o(nwords).neg()

        oloss = torch.bmm(ovectors, ivectors).squeeze().sigmoid().log().mean(1)
        nloss = torch.bmm(nvectors, ivectors).squeeze().sigmoid().log().view(-1, context_size, self.n_negs).sum(2).mean(1)
        return -(oloss + nloss).mean()