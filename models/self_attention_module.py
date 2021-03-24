import torch
import torch.nn as nn

from models.common import get_norm

import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 8, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 8, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x = x.permute(0, 2, 1)
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        #attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        x = x.permute(0, 2, 1)
        
        return x
    
    
class SelfAttention2(nn.Module):
    def __init__(self, channels):
        super(SelfAttention2, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 8, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 8, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias
        self.gamma = nn.Parameter(torch.tensor(0.0))

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x = x.permute(0, 2, 1)
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attn = self.softmax(energy)
        #attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        attn = torch.bmm(x_v, attn)
        #x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = self.gamma * attn + x
        x = x.permute(0, 2, 1)
        
        return x
    
    
class SelfAttentionT(nn.Module):
    def __init__(self, channels):
        super(SelfAttentionT, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias
        self.gamma = nn.Parameter(torch.tensor(0.0))

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x = x.permute(0, 2, 1)
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attn = self.softmax(energy)
        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attn)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = self.gamma * x_r + x
        x = x.permute(0, 2, 1)
        
        return x


class SelfAttention3(nn.Module):
    def __init__(self, channels):
        super(SelfAttention3, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 8, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 8, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias
        self.gamma = nn.Parameter(torch.tensor(0.0))

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x = x.permute(0, 2, 1)
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attn = self.softmax(energy)
        #attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        #attn = torch.bmm(x_v, attn)
        
        #x = self.gamma * attn + x
        #x = x.permute(0, 2, 1)
        
        return attn
    
    
class SelfAttentionME(nn.Module):
    def __init__(self, channels, bn_momentum=0.1, D=3):
        super(SelfAttentionME, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 8, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 8, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias
        self.gamma = nn.Parameter(torch.tensor(0.0))

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        #f = x.decomposed_features
        # features is a list of (n_points, feature_size) tensors with variable number of points
        #batch_size = len(f)
        
        f = x.feats
        shape = x.shape
        
        f = f.view((1, shape[1], shape[0]))
        x_q = self.q_conv(f).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(f)
        x_v = self.v_conv(f)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attn = self.softmax(energy)
        # b, c, n
        attn = torch.bmm(x_v, attn)
        x_att = self.gamma * attn + f
        x_att = x_att.view((x.shape[0], x.shape[1]))
        
        x = ME.SparseTensor(x_att, coords=x.coords)
        
        return x
    

class SelfAttention4(nn.Module):
    def __init__(self, channels):
        super(SelfAttention4, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias
        self.gamma = nn.Parameter(torch.tensor(0.0))

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x = x.permute(0, 2, 1)
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attn = self.softmax(energy)
        #attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        attn = torch.bmm(x_v, attn)
        #x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x_att = self.gamma * attn + x
        x_att = x_att.permute(0, 2, 1)
        
        x = torch.cat((x.permute(0, 2, 1), x_att), 1)
        
        return x
