import argparse
from pathlib import Path

from torch.xpu import temperature
from reasoning_from_scratch.utils import *
from reasoning_from_scratch.constants import *

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Run Qwen3 text generation")
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
        help="Kind of the model to download (e.g., 'base', 'reasoning').",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Max new tokens for generation",
    )
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
        "--cache",
        action='store_true',
        help="Use cache during generation. If provided, becomes True; otherwise False."
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

    model,tokenizer = load_model_and_tokenizer(args.kind, device, False, local_dir=str(downloads_dir))
    model.eval()

    if args.prompt is None:
        prompt = "Explain large language models in a single sentence."
    else:
        prompt = args.prompt

    print()
    print("=" * 60)
    print(f"torch     : {torch.__version__}")
    print(f"device    : {device}")
    print(f"cache     : {args.cache}")
    print("=" * 60)
    print()

    input_ids = tokenizer.encode(prompt)
    input_token_ids_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)
    # .unsqueeze(0) adds the batch dimension, so shape changes from: (seq_len,) to (1, seq_len)
    # the model sees a batch with exactly one sequence.

    start_time = time.time()
    all_token_ids = []

    if not args.cache:
        generate_fn = generate_text_basic_stream
    else:
        generate_fn = generate_text_basic_stream_cache

    for token in generate_fn(
        model=model,
        token_ids=input_token_ids_tensor,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    ):
        token_id = token.squeeze(0).item()
        print(tokenizer.decode([token_id]), end="", flush=True)
        all_token_ids.append(token_id)

    print("\n")

# change eos_token_id from None to tokenizer.eos_token_id
# Test script : py .\pipeline\2.generate_response.py --verbose --cache


# Inside the model, the first step is: embedding lookup.
# each token ID becomes a learned vector of size emb_dim
# if the prompt has N tokens, then x becomes: (1, N, emb_dim)
# Each transformer block does attention + MLP
    # the model builds a mask: This is a triangular mask that prevents each token from “looking at the future”.
    # MLP step
        # Then a feed-forward network is applied to each token independently
        # This adds nonlinear transformation so the model can represent more complex patterns.
# After all transformer blocks, each token has been transformed into a hidden state.
# For a prompt of length N: shape == (1, N, emb_dim) >> So for each position, there is a hidden vector of length emb_dim.
# The final linear layer is: nn.Linear(emb_dim, vocab_size)
# For one hidden vector, This is just a dot product between the hidden vector and each row of the weight matrix. So each vocabulary entry gets a score.

# Suppose: 
    # prompt length = 10 tokens
    # emb_dim = 4
    # vocabulary size = 5
    # hidden vector for the last prompt token: h = [0.8, -0.2, 1.1, 0.4]
    # output weights are : 
    # W = [
    #     [ 1.0,  0.2, -0.5,  0.1],   # token 0
    #     [ 0.3,  0.9,  0.2, -0.4],   # token 1
    #     [-0.8, 0.5,  1.7,  0.6],   # token 2
    #     [ 0.7, -0.1,  0.3,  1.2],   # token 3
    #     [ 2.0,  0.0, -0.2,  0.4],   # token 4
    # ]
    # logits = h @ W.T
    # Compute roughly:
        # token 0 :0.8∗1+(−0.2∗0.2)+1.1∗(−0.5)+0.4∗0.1=0.8−0.04−0.55+0.04=0.25
        # token 1: ...
        # token 2: ...
        # token 4: maybe highest
    
    # greedy decoding: Whichever token gets the biggest number is chosen as the next prediction. 
        # next_token = torch.argmax(out, dim=-1, keepdim=True)
    # temperature scaling: keep the logits distribution more or less sharp depending on temperature value. Lower temperature makes the distribution sharper, higher temperature makes it flatter.
        # logits = scale_logits_by_temperature(out, temperature)
        # probas = torch.softmax(logits, dim=-1)
        # next_token = torch.multinomial(probas.cpu(), num_samples=1)
    # top-p (nucleus) sampling: Keep the smallest set of tokens whose cumulative probability exceeds p; then sample from this set.
        # logits = scale_logits_by_temperature(out, temperature)
        # probas = torch.softmax(logits, dim=-1)
        # probas = top_p_filter(probas, top_p)
        # next_token = torch.multinomial(probas.cpu(), num_samples=1)
        # next_token = next_token.to(orig_device)
    
# In Layman's term
    # in the "output" layer of transformer, (its a logical layer and not actual)
    # input is context matrics - each vector of matric represent contextual relevance with each input token.
    # since we have applied causal masking (token can't look to future token), the last token vector would have the contextual reference of all input token
    # we multiply this vector with trained weight matrix to get logits - it represents probability of each vocab entry
    # so if assume model has 4 vocab words, and if logit is [0.2,0.3,0.4,0.1] - it says, the distribution of probabilities of each vocab entry
    # if we apply greedy search (choosing the highest probability) 3rd entry would be choosen as next token
