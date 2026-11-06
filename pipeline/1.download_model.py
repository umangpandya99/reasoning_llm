import argparse
from pathlib import Path

from reasoning_from_scratch.utils import download_model
from reasoning_from_scratch.constants import FILES


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--kind",
        type=str,
        default="base",
        help="Kind of the model to download (e.g., 'base', 'reasoning').",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Repository from which to download the model.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="model.safetensors",
        help="Model file name for the downloaded model. Use 'auto' to select automatically.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="tokenizer.json",
        help="Tokenizer file name for the downloaded model. Use 'auto' to select automatically.",
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    args.out_dir = str(repo_root / "downloads")

    files = {f"{args.kind}": {"model": f"{args.model}", "tokenizer": f"{args.tokenizer}"}}
    
    download_model(files, kind=args.kind, repo=args.repo, tokenizer_only=False, out_dir=args.out_dir)