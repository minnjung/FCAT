
import torch
import MinkowskiEngine as ME

from models.simple_resunet import ResUNetBN2C as simple_resunet
from models.netvlad import MinkNetVladWrapper
import layers.pooling as pooling

from models.self_attention_module import SelfAttention2 as SA

class FCAT(torch.nn.Module):
    def __init__(self, model, in_channels, feature_size, output_dim, planes, layers, num_top_down, conv0_kernel_size=5):
        super().__init__()
        self.model = model
        self.in_channels = in_channels
        self.feature_size = feature_size    # Size of local features produced by local feature extraction block
        self.output_dim = output_dim        # Dimensionality of the global descriptor
        
        self.backbone = simple_resunet(in_channels=in_channels,
                             out_channels=self.feature_size,
                             bn_momentum=0.05, normalize_feature=True, 
                             conv1_kernel_size=conv0_kernel_size)
        self.n_backbone_features = output_dim

        if model == 'FCAT_Max':
            assert self.feature_size == self.output_dim, 'output_dim must be the same as feature_size'
            self.pooling = pooling.MAC()
        elif model == 'FCAT_GeM':
            assert self.feature_size == self.output_dim, 'output_dim must be the same as feature_size'
            self.pooling = pooling.GeM()
        elif model == 'FCAT_GeM_ATT':
            assert self.feature_size == self.output_dim, 'output_dim must be the same as feature_size'
            self.SA = SA(self.feature_size).to()
            self.pooling = pooling.GeM_ATT()
        elif model == 'FCAT_NetVlad':
            self.pooling = MinkNetVladWrapper(feature_size=self.feature_size, 
                                              output_dim=self.output_dim,
                                              cluster_size=64, gating=False)
        elif model == 'FCAT_NetVlad_ATT':
            self.pooling = MinkNetVladWrapper(feature_size=self.feature_size, 
                                              output_dim=self.output_dim,
                                              cluster_size=64, gating=False, attn=True)
        elif model == 'FCAT_NetVlad_CG_ATT':
            self.pooling = MinkNetVladWrapper(feature_size=self.feature_size, 
                                              output_dim=self.output_dim,
                                              cluster_size=64, gating=True, attn=True)
        else:
            raise NotImplementedError('Model not implemented: {}'.format(model))

    def forward(self, batch):
        # Coords must be on CPU, features can be on GPU - see MinkowskiEngine documentation
        x = ME.SparseTensor(batch['features'], coords=batch['coords'])
        x = self.backbone(x)
        
        # x is (num_points, n_features) tensor
        assert x.shape[1] == self.feature_size, 'Backbone output tensor has: {} channels. Expected: {}'.format(x.shape[1], self.feature_size)
        
        x = self.pooling(x)
        
        assert x.dim() == 2, 'Expected 2-dimensional tensor (batch_size,output_dim). Got {} dimensions.'.format(x.dim())
        assert x.shape[1] == self.output_dim, 'Output tensor has: {} channels. Expected: {}'.format(x.shape[1], self.output_dim)
        # x is (batch_size, output_dim) tensor
        return x

    def print_info(self):
        print('Model class: MinkLoc')
        n_params = sum([param.nelement() for param in self.parameters()])
        print('Total parameters: {}'.format(n_params))
        n_params = sum([param.nelement() for param in self.backbone.parameters()])
        print('Backbone parameters: {}'.format(n_params))
        n_params = sum([param.nelement() for param in self.pooling.parameters()])
        print('Aggregation parameters: {}'.format(n_params))
        if hasattr(self.backbone, 'print_info'):
            self.backbone.print_info()
        if hasattr(self.pooling, 'print_info'):
            self.pooling.print_info()
