# Code taken from: https://github.com/filipradenovic/cnnimageretrieval-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME


class MAC(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = ME.MinkowskiGlobalMaxPooling()

    def forward(self, x: ME.SparseTensor):
        x = self.f(x)
        return x.F      # Return (batch_size, n_features) tensor


class SPoC(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = ME.MinkowskiGlobalAvgPooling()

    def forward(self, x: ME.SparseTensor):
        x = self.f(x)
        return x.F      # Return (batch_size, n_features) tensor


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.f = ME.MinkowskiGlobalAvgPooling()

    def forward(self, x: ME.SparseTensor):
        # This implicitly applies ReLU on x (clamps negative values)
        temp = ME.SparseTensor(x.F.clamp(min=self.eps).pow(self.p), coords=x.C)
        temp = self.f(temp)             # Apply ME.MinkowskiGlobalAvgPooling
        return temp.F.pow(1./self.p)    # Return (batch_size, n_features) tensor


class GeM_ATT(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM_ATT, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        b, n, c = x.shape
        return self.gem(x.permute(0, 2, 1), p=self.p, eps=self.eps).view(b, -1)
        
    def gem(self, x, p=3, eps=1e-6):
        #print(x.size(-2), x.size(-1))
        return F.avg_pool1d(x.clamp(min=eps).pow(p), x.size(-1)).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'