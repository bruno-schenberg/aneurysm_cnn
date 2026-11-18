"""Inspect nnU-Net checkpoint to extract encoder architecture details.

Loads a checkpoint from the multi-centric aneurysm nnU-Net model
(Zenodo 13386859, Dataset610, 3d_fullres) and prints:
  - State dict keys grouped by prefix (encoder vs decoder)
  - Encoder layer weight shapes and feature dimensions
  - Full architecture via model instantiation (if nnunetv2 available)
  - Parameter counts (encoder vs decoder breakdown)

Usage (on cluster):
    conda activate aneurysm_cnn_cuda
    python segmentation/inspect_nnunet.py
    python segmentation/inspect_nnunet.py --model-dir /custom/path/to/nnUNet_results
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch


DEFAULT_MODEL_DIR = Path.home() / "cnn_for_aneurysm/segmentation/models/nnUNet_results"


def _rglob_follow_symlinks(root: Path, pattern: str) -> list[Path]:
    """rglob that follows symlinks (Path.rglob doesn't by default)."""
    import glob as _glob
    return sorted(Path(p) for p in _glob.glob(str(root / "**" / pattern), recursive=True))


def find_checkpoints(model_dir: Path) -> list[Path]:
    """Find all .pth checkpoint files in the model directory tree."""
    return _rglob_follow_symlinks(model_dir, "*.pth")


def find_plans(model_dir: Path) -> list[Path]:
    """Find plans.json files that describe the network architecture."""
    return _rglob_follow_symlinks(model_dir, "plans.json")


def find_dataset_json(model_dir: Path) -> list[Path]:
    """Find dataset.json files."""
    return _rglob_follow_symlinks(model_dir, "dataset.json")


def print_separator(title: str = ""):
    print()
    print("=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)


def inspect_plans(plans_path: Path):
    """Parse and display architecture info from plans.json."""
    print_separator(f"PLANS: {plans_path}")

    with open(plans_path) as f:
        plans = json.load(f)

    # Print top-level keys
    print(f"\nTop-level keys: {list(plans.keys())}")

    # Network architecture is typically in configurations -> 3d_fullres
    if "configurations" in plans:
        for config_name, config in plans["configurations"].items():
            print(f"\n--- Configuration: {config_name} ---")
            arch_keys = [
                "architecture", "network_class_name",
                "UNet_class_name", "UNet_base_num_features",
                "n_conv_per_stage_encoder", "n_conv_per_stage_decoder",
                "num_pool_per_axis", "pool_op_kernel_sizes",
                "conv_kernel_sizes", "patch_size", "batch_size",
            ]
            for key in arch_keys:
                if key in config:
                    val = config[key]
                    print(f"  {key}: {json.dumps(val) if isinstance(val, (list, dict)) else val}")

            # Print all keys for completeness
            remaining = [k for k in config.keys() if k not in arch_keys]
            if remaining:
                print(f"  Other keys: {remaining}")

    # Also check for direct architecture fields (varies by nnU-Net version)
    for key in ["network_arch", "architecture", "network_class_name"]:
        if key in plans and key not in plans.get("configurations", {}).get("3d_fullres", {}):
            print(f"\n{key}: {plans[key]}")

    return plans


def group_state_dict_keys(state_dict: dict) -> dict[str, list[str]]:
    """Group state dict keys by their top-level prefix."""
    groups = defaultdict(list)
    for key in sorted(state_dict.keys()):
        prefix = key.split(".")[0]
        groups[prefix] = groups.get(prefix, [])
        groups[prefix].append(key)
    return dict(groups)


def classify_encoder_decoder(key: str) -> str:
    """Classify a state dict key as encoder, decoder, or other."""
    key_lower = key.lower()
    # nnU-Net typically uses 'encoder' and 'decoder' prefixes,
    # or 'down' / 'up' blocks, or stage numbering patterns
    encoder_patterns = ["encoder", "down", "input_block", "conv_blocks_context"]
    decoder_patterns = ["decoder", "up", "output_block", "conv_blocks_localization",
                        "seg_outputs", "tu"]  # tu = transpose upsampling

    for pat in encoder_patterns:
        if pat in key_lower:
            return "encoder"
    for pat in decoder_patterns:
        if pat in key_lower:
            return "decoder"
    return "other"


def inspect_checkpoint(ckpt_path: Path):
    """Load and inspect a single checkpoint file."""
    print_separator(f"CHECKPOINT: {ckpt_path.name}")
    print(f"Path: {ckpt_path}")
    print(f"Size: {ckpt_path.stat().st_size / 1e6:.1f} MB")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Print top-level keys in checkpoint
    if isinstance(ckpt, dict):
        print(f"\nCheckpoint top-level keys: {list(ckpt.keys())}")

        # Print non-tensor metadata
        for key in ckpt:
            if key in ("network_weights", "state_dict", "model_state_dict",
                       "optimizer_state", "optimizer_state_dict"):
                continue
            val = ckpt[key]
            if isinstance(val, (str, int, float, bool, type(None))):
                print(f"  {key}: {val}")
            elif isinstance(val, dict) and len(val) < 20:
                print(f"  {key}: {json.dumps(val, indent=4, default=str)}")
            elif isinstance(val, (list, tuple)) and len(val) < 20:
                print(f"  {key}: {val}")
            else:
                print(f"  {key}: <{type(val).__name__}, len={len(val) if hasattr(val, '__len__') else '?'}>")

        # Find the state dict
        state_dict = None
        for sd_key in ("network_weights", "state_dict", "model_state_dict", "model"):
            if sd_key in ckpt:
                state_dict = ckpt[sd_key]
                print(f"\nState dict found under key: '{sd_key}'")
                break

        if state_dict is None and all(isinstance(v, torch.Tensor) for v in list(ckpt.values())[:5]):
            state_dict = ckpt
            print("\nCheckpoint IS the state dict (no wrapper)")

        # Check for init_kwargs or architecture config stored in checkpoint
        for meta_key in ("init_kwargs", "init_args", "network_architecture",
                         "plans", "inference_allowed_mirroring_axes"):
            if meta_key in ckpt:
                val = ckpt[meta_key]
                print(f"\n--- {meta_key} ---")
                if isinstance(val, dict):
                    print(json.dumps(val, indent=2, default=str))
                else:
                    print(val)

    elif isinstance(ckpt, torch.nn.Module):
        print("\nCheckpoint is a full model object")
        state_dict = ckpt.state_dict()
    else:
        print(f"\nUnexpected checkpoint type: {type(ckpt)}")
        return None, None

    if state_dict is None:
        print("WARNING: Could not find state dict in checkpoint")
        return ckpt, None

    print(f"\nTotal keys in state dict: {len(state_dict)}")

    # Group keys by prefix
    print_separator("STATE DICT KEYS (grouped by top-level prefix)")
    groups = group_state_dict_keys(state_dict)
    for prefix, keys in sorted(groups.items()):
        print(f"\n  [{prefix}] ({len(keys)} parameters)")
        for k in keys[:10]:
            shape = tuple(state_dict[k].shape) if isinstance(state_dict[k], torch.Tensor) else "N/A"
            print(f"    {k:60s} {str(shape)}")
        if len(keys) > 10:
            print(f"    ... and {len(keys) - 10} more")

    # Encoder vs decoder classification
    print_separator("ENCODER vs DECODER CLASSIFICATION")
    enc_params, dec_params, other_params = 0, 0, 0
    enc_keys, dec_keys, other_keys = [], [], []

    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        n = tensor.numel()
        role = classify_encoder_decoder(key)
        if role == "encoder":
            enc_params += n
            enc_keys.append(key)
        elif role == "decoder":
            dec_params += n
            dec_keys.append(key)
        else:
            other_params += n
            other_keys.append(key)

    total = enc_params + dec_params + other_params
    print(f"\n  Encoder parameters:  {enc_params:>12,d}  ({100*enc_params/total:.1f}%)")
    print(f"  Decoder parameters:  {dec_params:>12,d}  ({100*dec_params/total:.1f}%)")
    print(f"  Other parameters:    {other_params:>12,d}  ({100*other_params/total:.1f}%)")
    print(f"  Total parameters:    {total:>12,d}")
    print(f"\n  Encoder layers: {len(enc_keys)}")
    print(f"  Decoder layers: {len(dec_keys)}")
    print(f"  Other layers:   {len(other_keys)}")

    # Show encoder weight shapes (conv layers only)
    print_separator("ENCODER CONV WEIGHT SHAPES (feature dimension progression)")
    conv_weights = []
    for key in enc_keys:
        tensor = state_dict[key]
        if isinstance(tensor, torch.Tensor) and "weight" in key and tensor.ndim >= 4:
            conv_weights.append((key, tuple(tensor.shape)))

    if conv_weights:
        for key, shape in conv_weights:
            if len(shape) == 5:
                print(f"  {key:60s}  out={shape[0]:4d}  in={shape[1]:4d}  kernel={shape[2]}x{shape[3]}x{shape[4]}")
            elif len(shape) == 4:
                print(f"  {key:60s}  out={shape[0]:4d}  in={shape[1]:4d}  kernel={shape[2]}x{shape[3]}")
    else:
        print("  No conv weights found in encoder keys.")
        print("  Showing all weight tensors with ndim >= 4:")
        for key, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor) and "weight" in key and tensor.ndim >= 4:
                shape = tuple(tensor.shape)
                role = classify_encoder_decoder(key)
                if len(shape) == 5:
                    print(f"  [{role:7s}] {key:55s}  out={shape[0]:4d}  in={shape[1]:4d}  kernel={shape[2]}x{shape[3]}x{shape[4]}")
                elif len(shape) == 4:
                    print(f"  [{role:7s}] {key:55s}  out={shape[0]:4d}  in={shape[1]:4d}  kernel={shape[2]}x{shape[3]}")

    # If 'other' keys are classified, show them for manual review
    if other_keys:
        print_separator("UNCLASSIFIED ('other') LAYERS")
        for key in other_keys[:30]:
            tensor = state_dict[key]
            shape = tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else "N/A"
            print(f"  {key:60s} {str(shape)}")
        if len(other_keys) > 30:
            print(f"  ... and {len(other_keys) - 30} more")

    return ckpt, state_dict


def try_instantiate_model(ckpt: dict, plans_path: Path | None):
    """Try to instantiate the nnU-Net model and print its architecture."""
    print_separator("MODEL INSTANTIATION")

    try:
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    except ImportError:
        print("  nnunetv2 not installed -- skipping model instantiation.")
        print("  Install with: pip install nnunetv2")
        return

    if plans_path is None:
        print("  No plans.json found -- cannot instantiate model.")
        return

    try:
        with open(plans_path) as f:
            plans = json.load(f)

        plans_manager = PlansManager(plans)
        config_name = "3d_fullres"

        if config_name not in plans_manager.available_configurations:
            print(f"  Config '{config_name}' not in plans. Available: {plans_manager.available_configurations}")
            return

        config_manager = plans_manager.get_configuration(config_name)

        # Get number of input channels and output classes from dataset.json
        dataset_json_path = plans_path.parent / "dataset.json"
        num_input_channels = 1
        num_classes = 2  # background + aneurysm
        if dataset_json_path.exists():
            with open(dataset_json_path) as f:
                ds = json.load(f)
            num_input_channels = len(ds.get("channel_names", {"0": "TOF"}))
            num_classes = len(ds.get("labels", {"background": 0, "aneurysm": 1}))
            print(f"  From dataset.json: input_channels={num_input_channels}, num_classes={num_classes}")

        # Try to build the network
        # nnU-Net v2 stores network architecture info in the plans
        network_class = config_manager.network_arch_class_name if hasattr(config_manager, 'network_arch_class_name') else None
        print(f"  Network class: {network_class}")

        # Use nnU-Net's own model builder
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
        model = get_network_from_plans(
            plans_manager,
            ds if dataset_json_path.exists() else {"channel_names": {"0": "TOF"}, "labels": {"background": 0, "aneurysm": 1}},
            config_name,
            num_input_channels=num_input_channels,
        )

        print(f"\n  Model class: {type(model).__name__}")
        print(f"\n--- Full architecture ---")
        print(model)

        # Count by module type
        print_separator("PARAMETER COUNT BY MODULE TYPE")
        type_counts = defaultdict(lambda: [0, 0])  # [n_modules, n_params]
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # leaf modules only
                n = sum(p.numel() for p in module.parameters())
                type_counts[type(module).__name__][0] += 1
                type_counts[type(module).__name__][1] += n

        for mtype, (count, params) in sorted(type_counts.items(), key=lambda x: -x[1][1]):
            if params > 0:
                print(f"  {mtype:30s}  x{count:3d}   {params:>12,d} params")

    except Exception as e:
        print(f"  Failed to instantiate model: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect nnU-Net checkpoint and extract encoder architecture details"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Path to nnUNet_results directory (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Which fold checkpoint to inspect in detail (default: 0)",
    )
    parser.add_argument(
        "--no-instantiate",
        action="store_true",
        help="Skip model instantiation (only inspect checkpoint weights)",
    )
    args = parser.parse_args()

    model_dir = args.model_dir
    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}")
        print("Provide --model-dir or ensure the default path exists.")
        return

    print(f"Model directory: {model_dir}")

    # Find all relevant files
    checkpoints = find_checkpoints(model_dir)
    plans_files = find_plans(model_dir)
    dataset_files = find_dataset_json(model_dir)

    print(f"\nFound {len(checkpoints)} checkpoint(s):")
    for cp in checkpoints:
        print(f"  {cp.relative_to(model_dir)}  ({cp.stat().st_size / 1e6:.1f} MB)")

    print(f"\nFound {len(plans_files)} plans.json file(s):")
    for p in plans_files:
        print(f"  {p.relative_to(model_dir)}")

    print(f"\nFound {len(dataset_files)} dataset.json file(s):")
    for d in dataset_files:
        print(f"  {d.relative_to(model_dir)}")

    # Inspect plans.json
    plans = None
    plans_path = None
    if plans_files:
        plans_path = plans_files[0]
        plans = inspect_plans(plans_path)

    # Inspect dataset.json
    if dataset_files:
        print_separator(f"DATASET: {dataset_files[0]}")
        with open(dataset_files[0]) as f:
            ds = json.load(f)
        for key in ("channel_names", "labels", "numTraining", "file_ending"):
            if key in ds:
                print(f"  {key}: {ds[key]}")

    # Pick one checkpoint to inspect in detail
    if not checkpoints:
        print("\nERROR: No checkpoint files found")
        return

    # Try to find the requested fold's checkpoint
    target_ckpt = None
    fold_str = f"fold_{args.fold}"
    for cp in checkpoints:
        if fold_str in str(cp) and "final" in cp.name:
            target_ckpt = cp
            break
    if target_ckpt is None:
        for cp in checkpoints:
            if fold_str in str(cp) and "best" in cp.name:
                target_ckpt = cp
                break
    if target_ckpt is None:
        for cp in checkpoints:
            if fold_str in str(cp):
                target_ckpt = cp
                break
    if target_ckpt is None:
        target_ckpt = checkpoints[0]
        print(f"\nWARNING: No fold_{args.fold} checkpoint found, using: {target_ckpt.name}")

    ckpt, state_dict = inspect_checkpoint(target_ckpt)

    # Try to instantiate the model
    if not args.no_instantiate and ckpt is not None:
        try_instantiate_model(ckpt, plans_path)

    print_separator("DONE")


if __name__ == "__main__":
    main()
