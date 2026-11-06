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
        "--device",
        type=str,
        default="cpu",
        help="Device to run on (e.g. 'cpu', 'cuda', 'mps'). "
            "If not provided, will auto-detect with get_device()."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Max new tokens for generation",
    )
    # If --verbose is provided, it becomes True. If omitted, it defaults to False.
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="Print per-sample correctness while evaluating. If provided, becomes True; otherwise False.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help=("Use a custom prompt. If not explicitly provided, uses the following defaults: "
            "'Explain large language models in a single sentence.' for the base model, and "
            "'Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.' for the reasoning model.")
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.35,
        help="Temperature for generation. Defaults to 0.35."
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.8,
        help="Top-p value for generation. Defaults to 0.8."
    ) 
    parser.add_argument(
        "--self_consistency",
        action='store_true',
        help="Enable self-consistency for generation. If provided, becomes True; otherwise False."
    ) 
    parser.add_argument(
            "--self_refinement",
            action='store_true',
            help="Enable self-refinement for generation. If provided, becomes True; otherwise False."
    )
    parser.add_argument(
        "--scorer",
        type=str,
        choices=["max_vote", "heuristic", "log_prob", "None"],
        default=None,
        help="Scoring method used to rank all generated answers and choose the best one."
    )

    args = parser.parse_args()

    if (args.self_consistency or args.self_refinement) and args.scorer is None:
        parser.error("The --scorer argument is required when --self_consistency or --self_refinement is provided. Choose one of: max_vote, heuristic, log_prob.")

    return args

if __name__ == "__main__":
    args = parse_args()   

    repo_root = Path(__file__).resolve().parents[1]
    downloads_dir = repo_root / "downloads"
    device = torch.device(args.device) if args.device else get_device()

    model,tokenizer = load_model_and_tokenizer(args.kind, device, False, local_dir=str(downloads_dir))
    model.eval()

    if args.prompt is None:
        raw_prompt = (
        "Half the value of $3x-9$ is $x+37$. "
        "What is the value of $x$?"
        )
    else:
        raw_prompt = args.prompt
    prompt = render_template(raw_prompt)

    print()
    print("=" * 60)
    print(f"torch     : {torch.__version__}")
    print(f"device    : {device}")
    print(f"cache     : {True}")
    print("=" * 60)
    print()

    if args.temperature and not args.top_p and not args.self_consistency:
        generate_fn = generate_text_temp_stream_cache
    elif args.top_p and not args.self_consistency:
        generate_fn = generate_text_top_p_stream_cache
    elif args.self_consistency:
        generate_fn = self_consistency_vote
    elif args.self_refinement:
        generate_fn = self_refinement_loop

    if args.self_consistency:
        results = self_consistency_vote(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            num_samples=5,
            # seed=None,
            # show_progress=True,
        )
        if args.scorer == "max_vote" or args.scorer == "None":
            print(f"best_answer: {results.get('final_answer', None)}")
            if results["final_answer"] is not None:
                extracted = results["final_answer"]

        elif args.scorer in {"heuristic", "log_prob"}:
            score_fn = {
                "heuristic": lambda answer: heuristic_score(answer),
                "log_prob": lambda answer: avg_logprob_answer(model, tokenizer, prompt, answer, device),
            }[args.scorer]

            score_by_idx = [
                (idx, cand, score_fn(results["full_answers"][idx]))
                for cand in results["majority_winners"]
                for idx in results["groups"][cand]
            ]
            if score_by_idx:
                best_idx, best, best_score = max(score_by_idx, key=lambda x: x)
                extracted = results["full_answers"][best_idx]
            else:
                best_idx = best = None
                best_score = float("-inf")
                extracted = None

            score_name = "heuristic" if args.scorer == "heuristic" else "log probability"
            print(f"Best answer based on {score_name} score: {extracted}")
    
    elif args.self_refinement:
        from functools import partial

        avg_logprob_score = partial(
            avg_logprob_answer,
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        results = self_refinement_loop(
            model=model,
            tokenizer=tokenizer,
            raw_prompt=raw_prompt,
            device=device,
            iterations=2,
            max_response_tokens=args.max_new_tokens,
            max_critique_tokens = 256,
            score_fn={
                "max_vote": None,
                "None": None,
                "heuristic": heuristic_score,
                "log_prob": avg_logprob_score,
            }.get(args.scorer, None),
            prompt_renderer=render_template,
            verbose=args.verbose,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"Self-refinement results:{results}")
        print(f"Final extracted answer: {results['final_full']}")

    else:
        generate_text_stream_concat_flex(
            model, tokenizer, prompt, device,
            max_new_tokens=args.max_new_tokens,
            verbose=args.verbose,
            generate_func=generate_fn,
            temperature=args.temperature,
            top_p=args.top_p,
        )

# Test script : 
# temperature and top_p sampling - Controlling randomness and diversity in generated outputs
# Inference with temprature and top_p sampling
# py .\pipeline\4.inference_scaling_self_refinement.py --verbose

# self-consistency - Generating N answers and selecting the best one based on the chosen scorer
# Inference with self-consistency and different scorers
# py .\pipeline\4.inference_scaling_self_refinement.py --verbose --self_consistency --scorer heuristic --temperature 0.8 --top_p 0.9
# py .\pipeline\4.inference_scaling_self_refinement.py --verbose --self_consistency --scorer log_prob --temperature 0.8 --top_p 0.9

# self-refinement - Iteratively refining the answer based on the chosen scorer
# Inference with self-refinement and different scorers
# py .\pipeline\4.inference_scaling_self_refinement.py --verbose --self_refinement --scorer None --temperature 0.7 --top_p 0.9
# py .\pipeline\4.inference_scaling_self_refinement.py --verbose --self_refinement --scorer heuristic --temperature 0.7 --top_p 0.9
# py .\pipeline\4.inference_scaling_self_refinement.py --verbose --self_refinement --scorer log_prob --temperature 0.7 --top_p 0.9