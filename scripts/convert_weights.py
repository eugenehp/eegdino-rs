#!/usr/bin/env python3
"""Convert EEG-DINO PyTorch .pt checkpoints to safetensors for eegdino-rs.

This script:
  1. Loads a PyTorch checkpoint (.pt file)
  2. Strips the 'module.student.' prefix from all keys
  3. Renames Sequential-indexed keys to descriptive names
  4. Transposes linear weight matrices from PyTorch [out, in] to Burn [in, out]
  5. Saves everything as float32 safetensors

Usage:
    python convert_weights.py \
        --input ../EEG-DINO/pre-trained-models/model_EEG_DINO_S.pt \
        --output weights/eeg_dino_small.safetensors

    # Or convert all three:
    python convert_weights.py --all \
        --input-dir ../EEG-DINO/pre-trained-models \
        --output-dir weights
"""

import argparse
import os
import sys

import torch
import numpy as np

try:
    from safetensors.torch import save_file
except ImportError:
    print("Please install safetensors: pip install safetensors", file=sys.stderr)
    sys.exit(1)


# Keys that are linear weights and need transposing (PyTorch [out, in] → Burn [in, out]).
# Conv2d weights do NOT need transposing (both use [out, in/groups, H, W]).
LINEAR_WEIGHT_PATTERNS = [
    # These match AFTER key remapping
    "spectral_proj.weight",     # Linear(101, d_model)
    "channel_embedding.weight", # Linear(19, d_model)
    "attn.qkv.weight",         # Linear(dim, 3*dim)
    "attn.proj.weight",        # Linear(dim, dim)
    "mlp.fc1.weight",          # Linear(dim, ffn_dim)
    "mlp.fc2.weight",          # Linear(ffn_dim, dim)
    # Classification head linears
    "full_linear.weight",
    "channel_linear.weight",
    "classifier.0.weight",
    "classifier.3.weight",
    "classifier.6.weight",
]

# Mapping from PyTorch Sequential indices to descriptive names
PROJ_IN_REMAP = {
    "0": "conv1",   # Conv2d
    "1": "norm1",   # GroupNorm
    # 2 = GELU (no params)
    "3": "conv2",   # Conv2d
    "4": "norm2",   # GroupNorm
    # 5 = GELU (no params)
    "6": "conv3",   # Conv2d
    "7": "norm3",   # GroupNorm
    # 8 = GELU (no params)
}

TIME_ENCODING_REMAP = {
    "0": "",  # Single Conv2d, drop the index
}

SPECTRAL_PROJ_REMAP = {
    "0": "",  # Single Linear, drop the index
    # 1 = Dropout (no params)
}


def is_linear_weight(key: str) -> bool:
    """Check if a key corresponds to a linear weight that needs transposing."""
    return any(key.endswith(pat) for pat in LINEAR_WEIGHT_PATTERNS)


def remap_key(key: str) -> str:
    """Remap a key from PyTorch naming to clean Rust naming.

    Handles:
      - proj_in.{0,1,3,4,6,7}.{weight,bias} → proj_in.{conv1,norm1,...}.{weight,bias}
      - time_encoding.0.{weight,bias} → time_encoding.{weight,bias}
      - spectral_proj.0.{weight,bias} → spectral_proj.{weight,bias}
    """
    # proj_in sequential indices
    if ".proj_in." in key:
        parts = key.split(".")
        for i, part in enumerate(parts):
            if parts[i - 1] == "proj_in" if i > 0 else False:
                if part in PROJ_IN_REMAP:
                    parts[i] = PROJ_IN_REMAP[part]
                break
        # Find proj_in index and remap
        for i in range(len(parts)):
            if parts[i] == "proj_in" and i + 1 < len(parts):
                idx = parts[i + 1]
                if idx in PROJ_IN_REMAP:
                    parts[i + 1] = PROJ_IN_REMAP[idx]
                break
        key = ".".join(p for p in parts if p)

    # time_encoding.0 → time_encoding
    if ".time_encoding.0." in key:
        key = key.replace(".time_encoding.0.", ".time_encoding.")

    # spectral_proj.0 → spectral_proj
    if ".spectral_proj.0." in key:
        key = key.replace(".spectral_proj.0.", ".spectral_proj.")

    return key


def convert_checkpoint(input_path: str, output_path: str):
    """Convert a single checkpoint."""
    print(f"Loading {input_path}...")
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)

    # Extract state dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    output = {}
    skipped = []

    for raw_key, tensor in state_dict.items():
        # Only keep student encoder weights (skip teacher, projectors, losses)
        if raw_key.startswith("module.student."):
            key = raw_key[15:]  # strip 'module.student.'
        elif raw_key.startswith("encoder."):
            # Finetuned model: encoder. prefix → keep sub-keys
            key = raw_key[8:]
        elif not raw_key.startswith("module."):
            # No prefix — use as-is (e.g. direct state_dict)
            key = raw_key
        else:
            # Skip module.teacher.*, module.mask_projector.*, etc.
            skipped.append(raw_key)
            continue

        # Skip relative position index (integer tensor, not a weight)
        if "relative_position_index" in key:
            skipped.append(raw_key)
            continue

        # Remap sequential indices to descriptive names
        key = remap_key(key)

        # Convert to float32
        tensor = tensor.float().contiguous()

        # Transpose linear weights: PyTorch [out, in] → Burn [in, out]
        if is_linear_weight(key) and tensor.dim() == 2:
            tensor = tensor.t().contiguous()
            print(f"  {raw_key} → {key}  {list(tensor.shape)}  (transposed)")
        else:
            print(f"  {raw_key} → {key}  {list(tensor.shape)}")

        output[key] = tensor

    if skipped:
        print(f"\nSkipped {len(skipped)} keys: {skipped}")

    # Detect model size
    if "global_tokens" in output:
        d_model = output["global_tokens"].shape[-1]
        size_name = {200: "Small", 512: "Medium", 1024: "Large"}.get(d_model, f"unknown(d={d_model})")
        print(f"\nDetected model size: {size_name} (d_model={d_model})")

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_file(output, output_path)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"Saved {len(output)} tensors to {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Convert EEG-DINO .pt to .safetensors")
    parser.add_argument("--input", type=str, help="Input .pt file")
    parser.add_argument("--output", type=str, help="Output .safetensors file")
    parser.add_argument("--all", action="store_true", help="Convert all three model sizes")
    parser.add_argument("--input-dir", type=str, default="../EEG-DINO/pre-trained-models",
                        help="Directory containing .pt files (with --all)")
    parser.add_argument("--output-dir", type=str, default="weights",
                        help="Output directory (with --all)")
    args = parser.parse_args()

    if args.all:
        models = [
            ("model_EEG_DINO_S.pt", "eeg_dino_small.safetensors"),
            ("model_EEG_DINO_M.pt", "eeg_dino_medium.safetensors"),
            ("model_EEG_DINO_L.pt", "eeg_dino_large.safetensors"),
        ]
        for inp, out in models:
            inp_path = os.path.join(args.input_dir, inp)
            out_path = os.path.join(args.output_dir, out)
            if os.path.exists(inp_path):
                convert_checkpoint(inp_path, out_path)
                print()
            else:
                print(f"Skipping {inp_path} (not found)")
    elif args.input and args.output:
        convert_checkpoint(args.input, args.output)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
