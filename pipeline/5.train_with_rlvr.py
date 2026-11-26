import argparse
from pathlib import Path
from reasoning_from_scratch.utils import *
from reasoning_from_scratch.constants import *

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Run Qwen3 text generation")
    parser.add_argument(
        "--kind",
        type=str,
        default="base",
        help="Kind of the model to download (e.g., 'base', 'reasoning').",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of training steps",
    )
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=4,
        help="Number of rollouts for training",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max new tokens for generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature for generation. Defaults to 0.8."
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p value for generation. Defaults to 0.9."
    ) 
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=5,
        help="Checkpoint every N steps. Defaults to 5."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="model_checkpoint",
        help="Directory to save checkpoints. Defaults to 'model_checkpoint'."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate for training. Defaults to 1e-5."
    )
    

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()   

    repo_root = Path(__file__).resolve().parents[1]
    downloads_dir = repo_root / "downloads"

    device = get_device()
    model,tokenizer = load_model_and_tokenizer(args.kind, device, False, local_dir=str(downloads_dir))
    data = load_math_data(file="math_train_12000.json")

    train_rlvr_grpo(
        model=model,
        tokenizer=tokenizer,
        math_data=data,
        device=device,
        steps=args.steps,
        num_rollouts=args.num_rollouts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        lr=args.lr,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
    )