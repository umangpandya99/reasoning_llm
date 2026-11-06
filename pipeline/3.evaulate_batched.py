import argparse
import json
import time
from pathlib import Path
from reasoning_from_scratch.utils import extract_final_candidate,grade_answer,eta_progress_message,get_device,load_math500_data
# from reasoning_from_scratch.constants import *
from reasoning_from_scratch.qwen3_batched import *

def render_prompt(prompt):
    template = (
        "You are a helpful math assistant.\n"
        "Answer the question and write the final result on a new line as:\n"
        "\\boxed{ANSWER}\n\n"
        f"Question:\n{prompt}\n\nAnswer:"
    )
    return template


def evaluate_math500_batched(
    model,
    tokenizer,
    device,
    math_data,
    out_path=None,
    max_new_tokens=512,
    verbose=False,
    batch_size=4,
    show_eta=False,
):
    model.eval()

    # Default output path like the streaming variant
    if out_path is None:
        dev_name = str(device).replace(":", "-")  # Make filename compatible with Windows
        out_path = Path(f"math500-{dev_name}.jsonl")

    num_examples = len(math_data)
    num_correct = 0

    start_time = time.time()

    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)

    with open(out_path, "w", encoding="utf-8") as f:
        # Batched loop
        for start in range(0, num_examples, batch_size):
            batch = math_data[start:start + batch_size]

            prompts = [render_prompt(row["problem"]) for row in batch]

            # Encode and left-pad
            tokenized = [tokenizer.encode(p) for p in prompts]
            max_len = max(len(t) for t in tokenized)
            left_padded = [
                ([pad_id] * (max_len - len(t)) + t) if pad_id is not None else t
                for t in tokenized
            ]
            input_ids = torch.tensor(left_padded, device=device, dtype=torch.long)

            # Generate (batched)
            gen = generate_text_basic_batched_cache(
                model,
                token_ids=input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_id,
                pad_id=pad_id,
            )  # shape: (B, T_new)

            # Process each row
            B = gen.size(0)
            for b_idx in range(B):
                # Match the variable names used by evaluate_math500_stream
                i = start + b_idx + 1
                row = batch[b_idx]

                row_tokens = gen[b_idx]
                # Cut at first EOS if present
                if eos_id is not None:
                    eos_pos = (row_tokens == eos_id).nonzero(as_tuple=True)[0]
                    if len(eos_pos) > 0:
                        row_tokens = row_tokens[: eos_pos[0]]

                gen_text = tokenizer.decode(row_tokens.tolist())
                extracted = extract_final_candidate(gen_text)
                is_correct = grade_answer(extracted, row["answer"])
                num_correct += int(is_correct)

                record = {  # Record to be saved for inspection
                    "index": i,
                    "problem": row["problem"],
                    "gtruth_answer": row["answer"],
                    "generated_text": gen_text,
                    "extracted": extracted,
                    "correct": bool(is_correct),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                progress_msg = eta_progress_message(
                    processed=i,
                    total=num_examples,
                    start_time=start_time,
                    show_eta=show_eta,
                    label="MATH-500",
                )
                print(progress_msg, end="\r", flush=True)
                if verbose:  # Print responses during the generation process
                    print(
                        f"\n\n{'='*50}\n{progress_msg}\n"
                        f"{'='*50}\nExtracted: {extracted}\n"
                        f"Expected:  {row['answer']}\n"
                        f"Correct so far: {num_correct}\n{'-'*50}"
                    )

    # Print summary information
    seconds_elapsed = time.time() - start_time
    acc = num_correct / num_examples if num_examples else 0.0
    print(f"\nAccuracy: {acc*100:.1f}% ({num_correct}/{num_examples})")
    print(f"Total time: {seconds_elapsed/60:.1f} min")
    print(f"Logs written to: {out_path}")
    return num_correct, num_examples, acc

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use: 'auto', or any torch device string like 'cpu', 'cuda', 'cuda:0', 'mps'.",
    )
    parser.add_argument(
        "--kind",
        type=str,
        default="base",
        choices=["base", "reasoning", "instruct"],
        help="Model variant to load",
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=10,
        help="Number of MATH-500 examples to evaluate",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Max new tokens for generation",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for the model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for batched generation",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample correctness while evaluating. If provided, becomes True; otherwise False.",
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    downloads_dir = repo_root / "downloads"
    
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)

    print("Model:", args.kind)
    print("Device:", device)
    dev_name = str(device).replace(":", "-")
    print("Batch size:", args.batch_size)

    math_data = load_math500_data()[:args.dataset_size]

    model, tokenizer = load_model_and_tokenizer(args.kind, device, use_compile=args.compile, local_dir=str(downloads_dir))

    out_path = f"math500_{args.kind}-{dev_name}-batched-bs{args.batch_size}.jsonl"
    num_correct, num_examples, acc = evaluate_math500_batched(
        model=model,
        tokenizer=tokenizer,
        device=device,
        math_data=math_data,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        out_path=out_path,
        verbose=False,
        show_eta=True,
    )

# Test script : py .\pipeline\3.evaulate_batched.py --verbose