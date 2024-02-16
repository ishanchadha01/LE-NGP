'''
-----------------------------------------------------------------------------
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

import torch
from functools import partial
import numpy as np
import tinycudann as tcnn

from projects.neuralangelo.utils.spherical_harmonics import get_spherical_harmonics
from projects.neuralangelo.utils.misc import get_activation
from projects.nerf.utils import nerf_util


class LightLocEncodedRGB(torch.nn.Module):

    def __init__(self, cfg_rgb, feat_dim, appear_embed):
        super().__init__()
        self.cfg_rgb = cfg_rgb
        self.cfg_appear_embed = appear_embed
        encoding_view_dim = self.build_encoding(cfg_rgb.dir_encoding_view, cfg_rgb.lightloc_encoding_view)
        input_base_dim = 6 if cfg_rgb.mode == "idr" else 3
        input_dim = input_base_dim + encoding_view_dim + feat_dim + (appear_embed.dim if appear_embed.enabled else 0)
        self.build_mlp(cfg_rgb.mlp, input_dim=input_dim)

    def build_encoding(self, cfg_dir_encoding_view, cfg_lightloc_encoding_view):
        if cfg_dir_encoding_view.type == "fourier":
            encoding_view_dim = 6 * cfg_dir_encoding_view.levels
        elif cfg_dir_encoding_view.type == "spherical":
            self.spherical_harmonic_encoding = partial(get_spherical_harmonics, levels=cfg_dir_encoding_view.levels)
            encoding_view_dim = (cfg_dir_encoding_view.levels + 1) ** 2
        else:
            raise NotImplementedError("Unknown dir encoding type")
        
        if cfg_lightloc_encoding_view.type == "fourier":
            encoding_view_dim += 6 * cfg_lightloc_encoding_view.levels
        elif cfg_lightloc_encoding_view.type == "spherical":
            self.spherical_harmonic_encoding = partial(get_spherical_harmonics, levels=cfg_lightloc_encoding_view.levels)
            encoding_view_dim += (cfg_lightloc_encoding_view.levels + 1) ** 2
        else:
            raise NotImplementedError("Unknown dir encoding type")
        
        return encoding_view_dim

    def build_mlp(self, cfg_mlp, input_dim=3):
        # RGB prediction
        layer_dims = [input_dim] + [cfg_mlp.hidden_dim] * cfg_mlp.num_layers + [3]
        activ = get_activation(cfg_mlp.activ, **cfg_mlp.activ_params)
        self.mlp = nerf_util.MLPwithSkipConnection(layer_dims, skip_connection=cfg_mlp.skip, activ=activ,
                                                   use_weightnorm=cfg_mlp.weight_norm)

    def forward(self, points_3D, normals, rays_unit, center, feats, app):
        view_enc = self.encode_view(rays_unit, center)  # [...,LD]
        input_list = [points_3D, view_enc, normals, feats]
        if app is not None:
            input_list.append(app)
        if self.cfg_rgb.mode == "no_view_dir":
            input_list.remove(view_enc)
        if self.cfg_rgb.mode == "no_normal":
            input_list.remove(normals)
        input_vec = torch.cat(input_list, dim=-1)
        rgb = self.mlp(input_vec).sigmoid_()
        return rgb  # [...,3]

    def encode_view(self, rays_unit, center):
        if self.cfg_rgb.dir_encoding_view.type == "fourier":
            view_enc = nerf_util.positional_encoding(rays_unit, num_freq_bases=self.cfg_rgb.dir_encoding_view.levels)
        elif self.cfg_rgb.dir_encoding_view.type == "spherical":
            view_enc = self.spherical_harmonic_encoding(rays_unit)
        else:
            raise NotImplementedError("Unknown encoding type")
        
        # rays is in [B,R,N,3], center is [B,R,3], so we add dim
        N = rays_unit.shape[2]
        center = center.unsqueeze(2).repeat(1,1,N,1) #TODO: this is doubling size of input, instead we could just make 3rd dim N+1
        if self.cfg_rgb.lightloc_encoding_view.type == "fourier":
            view_enc_light = nerf_util.positional_encoding(center, num_freq_bases=self.cfg_rgb.lightloc_encoding_view.levels)
            view_enc = torch.cat((view_enc, view_enc_light), dim=-1) # concat along last dim
        elif self.cfg_rgb.lightloc_encoding_view.type == "spherical":
            view_enc_light = self.spherical_harmonic_encoding(center)
            view_enc = torch.cat((view_enc, view_enc_light), dim=-1) # concat along last dim
        else:
            raise NotImplementedError("Unknown encoding type")
        
        return view_enc
