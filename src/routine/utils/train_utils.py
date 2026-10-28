import torch
def _rank0_print(pl_module, msg: str):
        if getattr(pl_module.trainer, "is_global_zero", True):
            print(msg)
            
import hashlib
def stable_hash63(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).digest()  # 16 bytes
    v = int.from_bytes(h[:8], "big", signed=False)  # 0..2^64-1
    v = v & ((1 << 63) - 1)
    return v


def _top_model_module_names(keys, limit=20):
    seen = []
    seen_set = set()

    for key in keys:
        key = str(key)
        parts = key.split(".")
        module_name = f"model.{parts[0]}" if len(parts) > 0 and parts[0] else f"model.{key}"
        if module_name not in seen_set:
            seen.append(module_name)
            seen_set.add(module_name)

    return seen[:limit], max(0, len(seen) - limit)


def _print_key_list(title, keys, verbose):
    if not keys:
        return

    if verbose is None:
        shown = list(keys)
    else:
        shown = list(keys[:verbose])

    print(f"[ckpt] --- {title} ({len(keys)}) ---")
    for key in shown:
        print(f"  - {key}")

    if verbose is not None and len(keys) > verbose:
        print(f"  - ... and {len(keys) - verbose} more")

def load_weights_non_strict(lit_module, ckpt_path: str, verbose: int | None = 20):
    """Load weights only and skip mismatched shapes."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # Load weights into lit_module.model only.
    model_sd = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
    if len(model_sd) == 0:
        # Fallback for raw model state_dict checkpoints.
        model_sd = state_dict

    cur_sd = lit_module.model.state_dict()

    filtered_sd = {}
    skipped_shape = []
    skipped_missing = []

    for k, v in model_sd.items():
        if k not in cur_sd:
            skipped_missing.append(k)
            continue
        if cur_sd[k].shape != v.shape:
            skipped_shape.append((k, tuple(v.shape), tuple(cur_sd[k].shape)))
            continue
        filtered_sd[k] = v

    missing, unexpected = lit_module.model.load_state_dict(filtered_sd, strict=False)

    print(f"[ckpt] loaded weights (filtered strict=False) from: {ckpt_path}")
    print(f"[ckpt] loaded={len(filtered_sd)} "
          f"skipped_shape={len(skipped_shape)} skipped_missing={len(skipped_missing)} "
          f"missing={len(missing)} unexpected={len(unexpected)}")

    if skipped_shape:
        detail = [
            f"{k}: ckpt{shp_ckpt} != cur{shp_cur}"
            for k, shp_ckpt, shp_cur in skipped_shape
        ]
        _print_key_list("skipped (shape mismatch)", detail, verbose)

    if skipped_missing:
        _print_key_list("skipped (not in current model)", skipped_missing, verbose)

    if unexpected:
        _print_key_list("unexpected", unexpected, verbose)

    if missing:
        _print_key_list("missing", missing, verbose)
