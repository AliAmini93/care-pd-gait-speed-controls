"""Portable analysis kernel. See docs/provenance_protocol.md for scope, provenance and I/O."""
from __future__ import annotations
import sys
from argparse import Namespace
from pathlib import Path
import numpy as np
import torch
from torch import nn
from vendor_momask_encdec import Decoder, Encoder
from vendor_momask_residual_vq import ResidualVQ
OPT = {'num_quantizers': 6, 'shared_codebook': False, 'quantize_dropout_prob': 0.2, 'nb_code': 512, 'code_dim': 512, 'down_t': 2, 'stride_t': 2, 'width': 512, 'depth': 3, 'dilation_growth_rate': 3, 'vq_act': 'relu', 'vq_norm': None, 'mu': 0.99}
INPUT_WIDTH = 263
DOWNSAMPLE = 4
ROOT_CHANNELS = (0, 1, 2, 3)
MIN_FRAMES = 8

class RVQVAE(nn.Module):

    def __init__(self):
        super().__init__()
        o = OPT
        self.encoder = Encoder(INPUT_WIDTH, o['code_dim'], o['down_t'], o['stride_t'], o['width'], o['depth'], o['dilation_growth_rate'], activation=o['vq_act'], norm=o['vq_norm'])
        self.decoder = Decoder(INPUT_WIDTH, o['code_dim'], o['down_t'], o['stride_t'], o['width'], o['depth'], o['dilation_growth_rate'], activation=o['vq_act'], norm=o['vq_norm'])
        self.quantizer = ResidualVQ(num_quantizers=o['num_quantizers'], shared_codebook=o['shared_codebook'], quantize_dropout_prob=o['quantize_dropout_prob'], quantize_dropout_cutoff_index=0, nb_code=o['nb_code'], code_dim=o['code_dim'], args=Namespace(mu=o['mu']))

    @torch.no_grad()
    def representation(self, x):
        h = self.encoder(x.permute(0, 2, 1).float())
        return self.quantizer(h, sample_codebook_temp=0.5, return_all_codes=True)[0]

def expected_keys():
    return sorted(RVQVAE().state_dict().keys())

def load_strict(state_dict):
    model = RVQVAE()
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

def normalize(raw, mean, std):
    return (np.asarray(raw, dtype=np.float32) - mean.astype(np.float32)) / std.astype(np.float32)

def variant(x_norm, mode):
    x = np.array(x_norm, dtype=np.float32, copy=True)
    root = list(ROOT_CHANNELS)
    if mode == 'full':
        return x
    if mode == 'no_root':
        x[:, root] = 0.0
        return x
    if mode == 'root_only':
        keep = x[:, root].copy()
        x[:] = 0.0
        x[:, root] = keep
        return x
    raise ValueError(mode)

@torch.no_grad()
def walk_embedding(model, x_norm, device='cpu'):
    t = x_norm.shape[0] // DOWNSAMPLE * DOWNSAMPLE
    if t < MIN_FRAMES:
        return None
    x = torch.from_numpy(np.ascontiguousarray(x_norm[:t], dtype=np.float32))[None].to(device)
    rep = model.representation(x)
    return rep.mean(dim=-1)[0].double().cpu().numpy()
