import sys
import torch
from pathlib import Path
from typing import Tuple

def load_tensor(path: str) -> torch.Tensor:
    t = torch.load(path, map_location="cpu")
    if isinstance(t, dict) and "tensor" in t:
        t = t["tensor"]
    if isinstance(t, (list, tuple)):
        t = t[0]
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{path} did not contain a tensor (got {type(t)})")
    return t.detach().cpu().float().contiguous()

def compare_tensors(a: torch.Tensor, b: torch.Tensor, rtol=1e-4, atol=1e-5) -> Tuple[bool, float, float]:
    if a.shape != b.shape:
        return False, float("inf"), float("inf")
    diff = (a - b).abs()
    print(diff)
    max_abs = diff.max().item()
    denom = max(a.abs().max().item(), b.abs().max().item(), 1e-12)
    rel = max_abs / denom
    ok = torch.allclose(a, b, rtol=rtol, atol=atol)
    return ok, rel, max_abs

def summarize(x: torch.Tensor) -> str:
    return f"shape={tuple(x.shape)} dtype={x.dtype} min={x.min().item():.3e} max={x.max().item():.3e} mean={x.mean().item():.3e} std={x.std().item():.3e}"

def main(argv):
    if len(argv) < 3 or len(argv) % 2 == 0:
        print("Usage: python compare_tensors.py <a1.pt> <b1.pt> [<a2.pt> <b2.pt> ...]")
        print("Examples:")
        print("  python compare_tensors.py norm_original.pt norm_rotate.pt")
        print("  python compare_tensors.py attn_original.pt attn_rotate.pt ls1_original.pt ls1_rotate.pt")
        sys.exit(1)

    for i in range(1, len(argv), 2):
        a_path, b_path = argv[i], argv[i+1]
        a = load_tensor(a_path)
        b = load_tensor(b_path)

        ok, rel, max_abs = compare_tensors(a, b)
        print(f"\nComparing: {Path(a_path).name} vs {Path(b_path).name}")
        print("A:", summarize(a))
        print("B:", summarize(b))
        print(f"allclose: {ok} (rtol=1e-4, atol=1e-5) | max_abs={max_abs:.3e} rel={rel:.3e}")

        if not ok and a.shape == b.shape:
            # Optional: print where the largest error is
            idx = (a - b).abs().view(-1).argmax().item()
            flat_a = a.view(-1); flat_b = b.view(-1)
            print(f"worst index: {idx} | a={flat_a[idx].item():.6f} b={flat_b[idx].item():.6f} | abs_err={(flat_a[idx]-flat_b[idx]).abs().item():.3e}")

if __name__ == "__main__":
    main(sys.argv)