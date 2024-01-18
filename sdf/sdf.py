import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn


class SDFNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.xyz_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.3819,
            },
        )

        self.sdf_net = tcnn.Network(
            n_input_dims=32,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )

    
    def forward(self, x):
        # x: [B, 3] where B is batsh size
        x = (x + 1) / 2 # change range from [-1, 1] to [0, 1]
        x = self.encoder(x)
        h = self.sdf_net(x)
        return h