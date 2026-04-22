#!/usr/bin/env python3
"""Numerical parity test: run EEG-DINO in Python and save outputs for comparison.

Generates a deterministic input signal, runs it through each model size,
and saves the encoder outputs to .safetensors files for comparison with Rust.
"""
import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn

# Add the EEG-DINO repo to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'EEG-DINO'))

from models.transformer import TransformerEncoderLayer, Attention, Mlp


def build_encoder(feature_size, num_heads, num_layers, dim_feedforward, embedding_module):
    """Build a standalone encoder matching EEGEncoder."""
    from models.eeg_encoder import EEGEncoder
    import argparse
    args = argparse.Namespace(
        feature_size=feature_size,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        num_global_tokens=1,
        global_token_layer=1,
    )
    # Swap embedding module
    encoder = EEGEncoder(args)
    return encoder


def load_pretrained(encoder, checkpoint_path):
    """Load pretrained weights into encoder."""
    from collections import OrderedDict
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict']
    new_dict = OrderedDict()
    for key, val in state_dict.items():
        if key.startswith('module.student.'):
            new_key = key[15:]  # strip 'module.student.'
            new_dict[new_key] = val
    encoder.load_state_dict(new_dict, strict=False)
    return encoder


def run_parity_test(model_name, feature_size, num_heads, num_layers, dim_feedforward,
                     checkpoint_path, output_dir):
    """Run a single model's parity test."""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}: d={feature_size}, h={num_heads}, L={num_layers}, ff={dim_feedforward}")
    print(f"{'='*60}")

    # Import the correct embedding module
    if feature_size == 200:
        from models.embedding_small import PatchEmbedding
    elif feature_size == 512:
        from models.embedding_medium import PatchEmbedding
    elif feature_size == 1024:
        from models.embedding_large import PatchEmbedding
    else:
        raise ValueError(f"Unknown feature_size: {feature_size}")

    # Build encoder manually (eeg_encoder.py hardcodes embedding_small import)
    import argparse
    args = argparse.Namespace(
        feature_size=feature_size,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        num_global_tokens=1,
        global_token_layer=1,
    )

    class EEGEncoder(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.patch_embedding = PatchEmbedding(d_model=args.feature_size)
            self.encoder_layers = nn.ModuleList([
                TransformerEncoderLayer(
                    d_model=args.feature_size,
                    nhead=args.num_heads,
                    dim_feedforward=args.dim_feedforward,
                ) for _ in range(args.num_layers)
            ])
            self.global_tokens = nn.Parameter(
                torch.randn(1, args.num_global_tokens, args.feature_size)
            )
            self.global_token_layer = args.global_token_layer

        def forward(self, x_in):
            B, C, P, L = x_in.shape
            x = self.patch_embedding(x_in)
            b = x.shape[0]
            x = x.reshape(b, -1, x.shape[-1])
            global_tokens = self.global_tokens.expand(b, -1, -1)
            for i, encoder_layer in enumerate(self.encoder_layers):
                x = encoder_layer(x)
                if i + 1 == self.global_token_layer:
                    x = torch.cat([global_tokens, x], dim=1)
            return x

    encoder = EEGEncoder(args)
    encoder = load_pretrained(encoder, checkpoint_path)
    encoder.eval()

    # Monkey-patch the embedding forward to avoid .cuda() calls (run on CPU)
    device = torch.device('cpu')

    # Patch the PatchEmbedding forward to replace .cuda() with device
    original_forward = encoder.patch_embedding.forward
    def patched_forward(x):
        # The original forward calls torch.arange(...).cuda()
        # We intercept by temporarily replacing torch.Tensor.cuda
        original_cuda = torch.Tensor.cuda
        torch.Tensor.cuda = lambda self, *args, **kwargs: self
        try:
            return original_forward(x)
        finally:
            torch.Tensor.cuda = original_cuda
    encoder.patch_embedding.forward = patched_forward

    # Generate deterministic input
    torch.manual_seed(42)
    np.random.seed(42)

    # Input: 1 batch, 19 channels, 10 patches, 200 samples per patch
    B, C, P, L = 1, 19, 10, 200
    x = torch.randn(B, C, P, L) * 0.01  # small values, like normalized EEG

    # Save the input (before /100 normalization that happens in Rust)
    # Rust's encode_raw does x/100, so we save x*100 as the raw input.
    raw_input = (x * 100.0)

    with torch.no_grad():
        output = encoder(x)

    output_cpu = output

    print(f"  Input shape:  {list(x.shape)}")
    print(f"  Output shape: {list(output_cpu.shape)}")
    print(f"  Output[0,:5,:5]:")
    print(output_cpu[0, :5, :5])
    print(f"  Output min={output_cpu.min().item():.6f}, max={output_cpu.max().item():.6f}, "
          f"mean={output_cpu.mean().item():.6f}")

    # Save input and output
    from safetensors.torch import save_file
    os.makedirs(output_dir, exist_ok=True)
    save_file({
        'input': raw_input.float().contiguous(),
        'output': output_cpu.float().contiguous(),
    }, os.path.join(output_dir, f'parity_{model_name}.safetensors'))

    print(f"  Saved to {output_dir}/parity_{model_name}.safetensors")
    return output_cpu


def main():
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'tests', 'parity_data')
    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'EEG-DINO', 'pre-trained-models')

    models = [
        ('small', 200, 8, 12, 512, 'model_EEG_DINO_S.pt'),
        ('medium', 512, 16, 16, 1024, 'model_EEG_DINO_M.pt'),
        ('large', 1024, 16, 24, 2048, 'model_EEG_DINO_L.pt'),
    ]

    for name, fs, nh, nl, ff, ckpt_name in models:
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"Skipping {name}: {ckpt_path} not found")
            continue
        run_parity_test(name, fs, nh, nl, ff, ckpt_path, output_dir)


if __name__ == '__main__':
    main()
