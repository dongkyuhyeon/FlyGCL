"""Arithmetic lower bounds for persistent classifier statistics, not peak VRAM."""
import argparse
import json


def estimate(classes, expansion, feature_dim):
    # AIR/GACL official analytic statistics use float64; SLDA uses float32.
    values = {
        "AIR_class_A_only": classes * expansion * expansion * 8,
        "GACL_R_only": expansion * expansion * 8,
        "SLDA_Sigma_Lambda_means_counts": (2 * feature_dim**2 + classes * feature_dim + classes) * 4,
    }
    return {k: {"bytes": v, "GiB": v / 2**30} for k, v in values.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", type=int, default=100)
    parser.add_argument("--expansion", type=int, default=8192)
    parser.add_argument("--feature-dim", type=int, default=768)
    args = parser.parse_args()
    if min(args.classes, args.expansion, args.feature_dim) <= 0:
        parser.error("all dimensions must be positive")
    print(json.dumps(estimate(args.classes, args.expansion, args.feature_dim), indent=2))
