import torch
import re

QWEN_CONFIG_06_B = {
    "vocab_size": 151_936,     # Vocabulary size
    "context_length": 40_960,  # Length originally used during training
    "emb_dim": 1024,           # Embedding dimension
    "n_heads": 16,             # Number of attention heads
    "n_layers": 28,            # Number of layers
    "hidden_dim": 3072,        # Size of intermediate dim in FeedForward
    "head_dim": 128,           # Size of the heads in GQA
    "qk_norm": True,           # Whether to normalize queries & keys in GQA
    "n_kv_groups": 8,          # Key-Value groups for GQA
    "rope_base": 1_000_000.0,  # The base in RoPE's "theta"
    "dtype": torch.bfloat16,   # Lower-precision dtype to reduce memory
}

repo = "Qwen/Qwen3-0.6B"
FILES = {
        "base": {"model": "model.safetensors", "tokenizer": "tokenizer.json"},
        "reasoning": {"model": "qwen3-0.6B-reasoning.pth", "tokenizer": "tokenizer-reasoning.json"},
    }

# FILES = {
#         "base": {"model": "qwen3-0.6B-base.pth", "tokenizer": "tokenizer-base.json"},
#         "reasoning": {"model": "qwen3-0.6B-reasoning.pth", "tokenizer": "tokenizer-reasoning.json"},
#     }
repo = "rasbt/qwen3-from-scratch"
hf_fmt = "https://huggingface.co/{repo}/resolve/main/{file}"
backup_root = "https://f001.backblazeb2.com/file/reasoning-from-scratch/qwen3-0.6B"


RE_NUMBER = re.compile(
 r"-?(?:\d+/\d+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)

LATEX_FIXES = [
 (r"\\left\s*", ""),
 (r"\\right\s*", ""),
 (r"\\,|\\!|\\;|\\:", ""),
 (r"\\cdot", "*"),
 (r"\u00B7|\u00D7", "*"),
 (r"\\\^\\circ", ""),
 (r"\\dfrac", r"\\frac"),
 (r"\\tfrac", r"\\frac"),
 (r"°", ""),
]

RE_SPECIAL = re.compile(r"<\|[^>]+?\|>")

SUPERSCRIPT_MAP = {
 "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
 "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
 "⁺": "+", "⁻": "-", "⁽": "(", "⁾": ")",
}
