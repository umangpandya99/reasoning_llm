from pathlib import Path
import sys
import requests
import json
import time
import warnings
from urllib.parse import urlparse
from collections import Counter
import torch
import math
from sympy import simplify
from sympy.parsing import sympy_parser as spp
from sympy.core.sympify import SympifyError
from sympy.polys.polyerrors import PolynomialError
from tokenize import TokenError
from datasets import load_dataset
from reasoning_from_scratch.constants import *
from reasoning_from_scratch.qwen3 import *

# Get the appropriate device for computation (GPU/CPU)
def get_device(enable_tensor_cores=True):
 if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA CUDA GPU")

    if enable_tensor_cores:
        major, minor = map(int, torch.__version__.split(".")[:2])
        if (major, minor) >= (2, 9):
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            torch.backends.cudnn.conv.fp32_precision = "tf32"
        else:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
 elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
 elif torch.xpu.is_available():
    device = torch.device("xpu")
    print("Using Intel GPU")
 else:
    device = torch.device("cpu")
    print("Using CPU")
 return device

# Error message for download failures
def _download_error_message(filename, url, primary_error, backup_url=None, backup_error=None):
    details = [f"Failed to download {filename}."]

    if primary_error is not None:
        details.append(
            f"Primary URL failed ({url}): "
            f"{type(primary_error).__name__}: {primary_error}"
        )

    if backup_url and backup_error is not None:
        details.append(
            f"Backup URL failed ({backup_url}): "
            f"{type(backup_error).__name__}: {backup_error}"
        )

    cert_or_proxy_issue = any(
        isinstance(err, (requests.exceptions.ProxyError, requests.exceptions.SSLError))
        for err in (primary_error, backup_error)
        if err is not None
    )
    if not cert_or_proxy_issue:
        lowered = " ".join(
            str(err).lower() for err in (primary_error, backup_error) if err is not None
        )
        cert_or_proxy_issue = any(
            keyword in lowered for keyword in ("certificate", "ssl", "tls", "proxy")
        )

    if cert_or_proxy_issue:
        details.append(
            "This can happen on work or school machines where a VPN, proxy, or "
            "antivirus tool intercepts HTTPS certificates."
        )

    details.append(
        "See the troubleshooting guide: "
        "https://github.com/rasbt/reasoning-from-scratch/blob/main/troubleshooting.md "
        "(especially the 'File Download Issues' section)."
    )
    return "\n".join(details)

# Download model files from primary and backup URLs
def download_model(files,kind="base", repo="Qwen/Qwen3-0.6B",tokenizer_only=False, out_dir="."):
    if kind not in files:
        raise ValueError("kind must be 'base' or 'reasoning'")
    
    targets = ["tokenizer"] if tokenizer_only else ["model", "tokenizer"]

    for key in targets:
        fname = files[kind][key]
        primary = hf_fmt.format(repo=repo, file=fname)
        backup = f"{backup_root}/{fname}"
        download_file(primary, out_dir=out_dir, backup_url=backup)

# Download a single file from a URL with optional backup URL
def download_file(url, out_dir=".", backup_url=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(urlparse(url).path).name
    dest = out_dir / filename

    def try_download(u):
        try:
            with requests.get(u, stream=True, timeout=30) as r:
                r.raise_for_status()
                size_remote = int(r.headers.get("Content-Length", 0))

                # Skip download if already complete
                if dest.exists() and size_remote and dest.stat().st_size == size_remote:
                    print(f"✓ {dest} already up-to-date")
                    return True, None

                # Download in 1 MiB chunks with progress display
                block = 1024 * 1024
                downloaded = 0
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=block):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if size_remote:
                            pct = downloaded * 100 // size_remote
                            sys.stdout.write(
                                f"\r{filename}: {pct:3d}% "
                                f"({downloaded // (1024*1024)} MiB / "
                                f"{size_remote // (1024*1024)} MiB)"
                            )
                            sys.stdout.flush()
                if size_remote:
                    sys.stdout.write("\n")
            return True, None
        except requests.RequestException as exc:
            return False, exc

    # Try main URL first
    success, primary_error = try_download(url)
    if success:
        return dest

    # Try backup URL if provided
    backup_error = None
    if backup_url:
        print(f"Primary URL ({url}) failed.\nTrying backup URL ({backup_url})...")
        success, backup_error = try_download(backup_url)
        if success:
            return dest

    message = _download_error_message(
        filename=filename,
        url=url,
        primary_error=primary_error,
        backup_url=backup_url,
        backup_error=backup_error,
    )
    raise RuntimeError(message) from (backup_error or primary_error)

# Load model and tokenizer
def load_model_and_tokenizer(
 which_model, device, use_compile, local_dir="../downloads"
):
    if which_model == "base":
        # download_model(FILES, kind="base", tokenizer_only=False, out_dir=local_dir)
        print(f"Loading base model from {Path(local_dir)}")
        tokenizer_path = Path(local_dir) / "tokenizer-base.json"
        model_path = Path(local_dir) / "qwen3-0.6B-base.pth"
        tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)
    
    elif which_model == "reasoning":
        # download_model(FILES, kind="reasoning", tokenizer_only=False, out_dir=local_dir)
        
        tokenizer_path = Path(local_dir) / "tokenizer-reasoning.json"
        model_path = Path(local_dir) / "qwen3-0.6B-reasoning.pth"

        tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path,apply_chat_template=True,add_generation_prompt=True,add_thinking=True,)
    else:
        raise ValueError(f"Invalid choice: which_model={which_model}")
    
    model = Qwen3Model(QWEN_CONFIG_06_B)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    
    if use_compile:
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        model = torch.compile(model)
    
    return model, tokenizer

# Render a prompt using a predefined template
def render_template(prompt):
    template = (
        "You are a helpful math assistant.\n"
        "Answer the question and write the final result on a new line as:\n"
        "\\boxed{ANSWER}\n\n"
        f"Question:\n{prompt}\n\nAnswer:"
    )
    return template

# Generate and print statistics for the output tokens and device memory usage
def generate_stats(output_token_ids, start_time,
                   end_time):
    # tokenizer is currently unused but retained for backward compatibility
    total_time = end_time - start_time
    print(f"\n\nTime: {total_time:.2f} sec")
    print(f"{round(output_token_ids.numel() / total_time,2)} tokens/sec")

    for name, backend in (("CUDA", getattr(torch, "cuda", None)),
                          ("XPU", getattr(torch, "xpu", None))):
        if backend is not None and backend.is_available():

            # Check whether we are actually using this backend
            device_type = output_token_ids.device.type
            if device_type != name.lower():
                warnings.warn(
                    f"{name} is available but tensors are on "
                    f"{device_type}. Memory stats may be 0."
                )
    
            # Synchronize if supported (important for async backends)
            if hasattr(backend, "synchronize"):
                backend.synchronize()
            
            max_mem_bytes = backend.max_memory_allocated()
            max_mem_gb = max_mem_bytes / (1024 ** 3)
            print(f"Max {name} memory allocated: {max_mem_gb:.2f} GB")
            backend.reset_peak_memory_stats()

def get_last_boxed(text):
    boxed_start_idx = text.rfind(r"\boxed")
    if boxed_start_idx == -1:
        return None
    current_idx = boxed_start_idx + len(r"\boxed")

    while current_idx < len(text) and text[current_idx].isspace():
        current_idx += 1

    if current_idx >= len(text) or text[current_idx] != "{":
        return None
    current_idx += 1
    brace_depth = 1
    content_start_idx = current_idx

    while current_idx < len(text) and brace_depth > 0:
        char = text[current_idx]
        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
        current_idx += 1

    if brace_depth != 0:
        return None

    return text[content_start_idx:current_idx-1]

def extract_final_candidate(text, fallback="number_then_full"):
    result = ""
    if text:
        boxed = get_last_boxed(text.strip())
    if boxed:
        result = boxed.strip().strip("$ ")
    elif fallback in ("number_then_full", "number_only"):
        m = RE_NUMBER.findall(text)
        if m:
            result = m[-1]
        elif fallback == "number_then_full":
            result = text
    return result

def normalize_text(text):
    if not text:
        return ""
    text = RE_SPECIAL.sub("", text).strip()

    match = re.match(r"^[A-Za-z]\s*[.:]\s*(.+)$", text)
    if match:
        text = match.group(1)
    text = re.sub(r"\^\s*\{\s*\\circ\s*\}", "", text)
    text = re.sub(r"\^\s*\\circ", "", text)
    text = text.replace("°", "")
    match = re.match(r"^\\text\{(?P<x>.+?)\}$", text)
    if match:
        text = match.group("x")
    text = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", text)

    for pat, rep in LATEX_FIXES:
        text = re.sub(pat, rep, text)

    def convert_superscripts(s, base=None):
        converted = "".join(
        SUPERSCRIPT_MAP[ch] if ch in SUPERSCRIPT_MAP else ch
        for ch in s)
        if base is None:
            return converted
        return f"{base}**{converted}"

    text = re.sub(
        r"([0-9A-Za-z\)\]\}])([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)",
        lambda m: convert_superscripts(m.group(2), base=m.group(1)),
        text,
    )
    text = convert_superscripts(text)

    text = text.replace("\\%", "%").replace("$", "").replace("%", "")
    text = re.sub(
        r"\\sqrt\s*\{([^}]*)\}",
        lambda match: f"sqrt({match.group(1)})",
        text,
    )
    text = re.sub(
        r"\\sqrt\s+([^\\\s{}]+)",
        lambda match: f"sqrt({match.group(1)})",
        text,
    )

    text = re.sub(
        r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        lambda match: f"({match.group(1)})/({match.group(2)})",
        text,
    )
    text = re.sub(
        r"\\frac\s+([^\s{}]+)\s+([^\s{}]+)",
        lambda match: f"({match.group(1)})/({match.group(2)})",
        text,
    )

    text = text.replace("^", "**")
    text = re.sub(
        r"(?<=\d)\s+(\d+/\d+)",
        lambda match: "+" + match.group(1),
        text,
    )

    text = re.sub(
        r"(?<=\d),(?=\d\d\d(\D|$))",
        "",
        text,
    )
    return text.replace("{", "").replace("}", "").strip().lower()


def sympy_parser(expr):
    if expr is None or len(expr) > 2000:
        return None
    try:
        return spp.parse_expr(
            expr,
            transformations=(
                *spp.standard_transformations,
                spp.implicit_multiplication_application,
            ),
            evaluate=True,
        )
    except (SympifyError, SyntaxError, TypeError, AttributeError,
            IndexError, TokenError, ValueError, PolynomialError):
        return None

def equality_check(expr_gtruth, expr_pred):
    if expr_gtruth == expr_pred:
        return True

    gtruth, pred = sympy_parser(expr_gtruth), sympy_parser(expr_pred)
    if gtruth is not None and pred is not None:
        try:
            return simplify(gtruth - pred) == 0
        except (SympifyError, TypeError):
            pass
    return False

def split_into_parts(text):
    result = [text]
    if text:
        if (
        len(text) >= 2
        and text[0] in "([" and text[-1] in ")]"
        and "," in text[1:-1]
        ):
            items = [p.strip() for p in text[1:-1].split(",")]
            if all(items):
                result = items
    else:
        result = []
    return result

def grade_answer(pred_text, gt_text):
    result = False
    if pred_text is not None and gt_text is not None:
        gt_parts = split_into_parts(
            normalize_text(gt_text)
        )
        pred_parts = split_into_parts(
            normalize_text(pred_text)
        )
        if (gt_parts and pred_parts
        and len(gt_parts) == len(pred_parts)):
            result = all(
                equality_check(gt, pred)
                for gt, pred in zip(gt_parts, pred_parts)
            )
    return result 

# Load the MATH-500 test dataset, either from a local file or by downloading it from Hugging Face
def load_math500_data(local_path="../downloads/math500_test.json", save_copy=True):
    local_path = Path(local_path)

    # 1. Load from local file if it exists
    if local_path.exists():
        with local_path.open("r", encoding="utf-8") as f:
            math_data = json.load(f)
    else:
        # 2. Download from Hugging Face
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        
        # Convert to a standard Python list of dicts
        math_data = dataset.to_list() if hasattr(dataset, "to_list") else list(dataset)
        
        # 3. Save a local copy if requested
        if save_copy:
            # Ensure the parent directory (e.g., '../downloads') actually exists
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Since math_data is now a list, use standard json.dump
            with local_path.open("w", encoding="utf-8") as f:
                json.dump(math_data, f, ensure_ascii=False, indent=4)
                
    return math_data

def eta_progress_message( 
    processed,
    total,
    start_time,
    show_eta=False,
    label="Progress",
    ):
    progress = f"{label}: {processed}/{total}"
    pad_width = len(f"{label}: {total}/{total} | ETA: 00h 00m 00s")
    if not show_eta or processed <= 0:
        return progress.ljust(pad_width)
    elapsed = time.time() - start_time
    
    if elapsed <= 0:
        return progress.ljust(pad_width)
    
    remaining = max(total - processed, 0)
    
    if processed:
        avg_time = elapsed / processed
        eta_seconds = avg_time * remaining
    else:
        eta_seconds = 0
    
    eta_seconds = max(int(round(eta_seconds)), 0)
    minutes, rem_seconds = divmod(eta_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        eta = f"{hours}h {minutes:02d}m {rem_seconds:02d}s"
    elif minutes:
        eta = f"{minutes:02d}m {rem_seconds:02d}s"
    else:
        eta = f"{rem_seconds:02d}s"
    
    message = f"{progress} | ETA: {eta}"
    return message.ljust(pad_width)

def evaluate_math500_stream(
        model,
        tokenizer,
        device,
        math_data,
        out_path=None,
        max_new_tokens=512,
        verbose = False
):
    # save to output path
    if out_path is None:
        dev_name = str(device).replace(":", "-")
        out_path = Path(f"math500-{dev_name}.jsonl")

    num_examples = len(math_data)
    num_correct = 0
    start_time = time.time()

    with open(out_path, "w", encoding="utf-8") as f:
        # render template , generate ansewer, extract answer and grade
        for i, example in enumerate(math_data, start=1):
            prompt = render_template(example['problem'])
            model_answer = generate_text_stream_concat_flex(
                model, tokenizer, prompt, device,
                max_new_tokens=max_new_tokens,
                verbose=verbose,
                generate_func=generate_text_basic_stream_cache,
            )
            extracted = extract_final_candidate(model_answer)
            is_correct = grade_answer(extracted, example['answer'])

            num_correct += int(is_correct)

            # save the current example's evaluation record to the output file
            record = {
            "index": i,
            "problem": example["problem"],
            "gtruth_answer": example["answer"],
            "generated_text": model_answer,
            "extracted": extracted,
            "correct": bool(is_correct),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # print progress for the current example
            progress_msg = eta_progress_message(
                processed=i,
                total=num_examples,
                start_time=start_time,
                show_eta=True,
                label="MATH-500",
            )
            print(progress_msg, end="\r", flush=True)

            if verbose:
                print(
                f"\n\n{'='*50}\n{progress_msg}\n"
                f"{'='*50}\nExtracted: {extracted}\n"
                f"Expected: {example['answer']}\n"
                f"Correct so far: {num_correct}\n{'-'*50}"
                )

        seconds_elapsed = time.time() - start_time
        acc = num_correct / num_examples if num_examples else 0.0
        print(f"\nAccuracy: {acc*100:.1f}% ({num_correct}/{num_examples})")
        print(f"Total time: {seconds_elapsed/60:.1f} min")
        print(f"Logs written to: {out_path}")
        
        return num_correct, num_examples, acc

# Basic text generation without caching
@torch.inference_mode()
def generate_text_basic_stream(model,token_ids,max_new_tokens,eos_token_id=None):
 model.eval()
 start_time = time.time()
 generated_tokens = []
 try:
    for _ in range(max_new_tokens):
        out = model(token_ids)[:, -1]
        next_token = torch.argmax(out, dim=-1, keepdim=True)
        
        if (eos_token_id is not None and torch.all(next_token == eos_token_id)):
            break
        generated_tokens.append(next_token)
        yield next_token

        token_ids = torch.cat([token_ids, next_token], dim=1)
 finally:
    output_token_ids = torch.cat(generated_tokens, dim=1) if generated_tokens else token_ids[:, :0]
    generate_stats(output_token_ids, start_time, time.time())

# Basic text generation with caching
@torch.inference_mode()
def generate_text_basic_stream_cache(model,token_ids,max_new_tokens,eos_token_id=None): 
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    start_time = time.time()
    generated_tokens = []

    try:
        out = model(token_ids, cache=cache)[:, -1]
        for _ in range(max_new_tokens):
            next_token = torch.argmax(out, dim=-1, keepdim=True)
        
            if (eos_token_id is not None and torch.all(next_token == eos_token_id)):
                break
        
            generated_tokens.append(next_token)
            yield next_token
            out = model(next_token, cache=cache)[:, -1]
    finally:
        output_token_ids = torch.cat(generated_tokens, dim=1) if generated_tokens else token_ids[:, :0]
        generate_stats(output_token_ids, start_time, time.time())

# A wrapper for streamed text generation using basic stream with caching
# Not useful since we have generate_text_stream_concat_flex
# def generate_text_stream_concat(
#  model, tokenizer, prompt, device, max_new_tokens,
#  verbose=False,
# ):
    
#     input_ids = torch.tensor(
#         tokenizer.encode(prompt), device=device
#     ).unsqueeze(0) 
#     generated_ids = []
    
#     for token in generate_text_basic_stream_cache(
#         model=model,
#         token_ids=input_ids,
#         max_new_tokens=max_new_tokens,
#         eos_token_id=tokenizer.eos_token_id
#     ):
#         next_token_id = token.squeeze(0)
#         generated_ids.append(next_token_id.item())
#         if verbose:
#             print(
#                 tokenizer.decode(next_token_id.tolist()),
#                 end="",
#                 flush=True
#             )

#     return tokenizer.decode(generated_ids) 

# Flexible text generation with optional custom generation function and additional keyword arguments
def generate_text_stream_concat_flex(
        model, tokenizer, prompt, device, max_new_tokens,
        verbose=False,
        generate_func=None,
        **generate_kwargs
    ):
    if generate_func is None:
        generate_func = generate_text_basic_stream_cache

    input_ids = torch.tensor(
    tokenizer.encode(prompt), device=device
    ).unsqueeze(0)
    generated_ids = []
    for token in generate_func(
        model=model,
        token_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        **generate_kwargs,
    ):
        next_token_id = token.squeeze(0)
        generated_ids.append(next_token_id.item())
        if verbose:
            print(
            tokenizer.decode(next_token_id.tolist()),
            end="",
            flush=True
            )
    return tokenizer.decode(generated_ids)

# scale the logits 
def scale_logits_by_temperature(logits, temperature):
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    return logits / temperature

# Text generation with temperature scaling and caching
@torch.inference_mode()
def generate_text_temp_stream_cache(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None,
    temperature=0.
):
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    start_time = time.time()
    generated_tokens = []

    try:
        out = model(token_ids, cache=cache)[:, -1]
        for _ in range(max_new_tokens):
        # NEW:
        ########################################
            orig_device = token_ids.device
            if temperature is None or temperature == 0.0:
                next_token = torch.argmax(out, dim=-1, keepdim=True)
            else:
                logits = scale_logits_by_temperature(out, temperature)
                probas = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probas.cpu(), num_samples=1)
                next_token = next_token.to(orig_device)
            #########################################

            if (eos_token_id is not None and torch.all(next_token == eos_token_id)):
                break

            generated_tokens.append(next_token)
            
            yield next_token
            out = model(next_token, cache=cache)[:, -1]       
    finally:
        output_token_ids = torch.cat(generated_tokens, dim=1) if generated_tokens else token_ids[:, :0]
        generate_stats(output_token_ids, start_time, time.time())

def top_p_filter(probas, top_p):
    if top_p is None or top_p >= 1.0:
        return probas
    
    sorted_probas, sorted_idx = torch.sort(probas, dim=1, descending=True)
    cumprobas = torch.cumsum(sorted_probas, dim=1) 
    prefix = cumprobas - sorted_probas
    keep = prefix < top_p
    keep[:, 0] = True
    
    kept_sorted = torch.where(
        keep, sorted_probas,
        torch.zeros_like(sorted_probas)
    )

    filtered = torch.zeros_like(probas).scatter(1, sorted_idx, kept_sorted)
    
    denom = torch.sum(filtered, dim=1, keepdim=True).clamp_min(1e-12)
    return filtered / denom

# Text generation with top-p sampling and caching
@torch.inference_mode()
def generate_text_top_p_stream_cache(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None,
    temperature=0.,
    top_p=None
):
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    start_time = time.time()
    generated_tokens = []
    try:

        out = model(token_ids, cache=cache)[:, -1]
        for _ in range(max_new_tokens):    
            orig_device = token_ids.device
            if temperature is None or temperature == 0.0:
                next_token = torch.argmax(out, dim=-1, keepdim=True)
            else:
                logits = scale_logits_by_temperature(out, temperature)
                probas = torch.softmax(logits, dim=-1)
                # NEW:
                #########################################
                probas = top_p_filter(probas, top_p)
                #########################################
                next_token = torch.multinomial(probas.cpu(), num_samples=1)
                next_token = next_token.to(orig_device)

            if (eos_token_id is not None and torch.all(next_token == eos_token_id)):
                break
            
            generated_tokens.append(next_token)
            
            yield next_token
            out = model(next_token, cache=cache)[:, -1]
    finally:
        output_token_ids = torch.cat(generated_tokens, dim=1) if generated_tokens else token_ids[:, :0]
        generate_stats(output_token_ids, start_time, time.time())

# Self-consistency voting mechanism for multiple generated answers
def self_consistency_vote(
    model, tokenizer, prompt, device,
    num_samples=5, temperature=0.8, top_p=0.9, max_new_tokens=2048,
    show_progress=True, show_long_answer=False, seed=None,
):
    full_answers, short_answers = [], []
    
    for i in range(num_samples):
        if seed is not None:
            torch.manual_seed(seed + i + 1)

        answer = generate_text_stream_concat_flex(
            model=model, tokenizer=tokenizer, prompt=prompt, device=device,
            max_new_tokens=max_new_tokens, verbose=show_long_answer,
            generate_func=generate_text_top_p_stream_cache,
            temperature=temperature, top_p=top_p,
        )

        short = extract_final_candidate(
        answer, fallback="number_then_full"
        )
        full_answers.append(answer)
        short_answers.append(short)
        if show_progress:
            print(f"[Sample {i+1}/{num_samples}] → {short!r}")
    
    counts = Counter(short_answers)
    groups = {s: [] for s in counts}
    for idx, s in enumerate(short_answers):
        groups[s].append(idx)
    
    mc = counts.most_common()
    if not mc:
        majority_winners, final_answer = [], None
    else:
        top_freq = mc[0][1]
        majority_winners = [s for s, f in mc if f == top_freq]
        final_answer = mc[0][0] if len(majority_winners) == 1 else None
    
    return {
    "full_answers": full_answers,
    "short_answers": short_answers,
    "counts": dict(counts),
    "groups": groups,
    "majority_winners": majority_winners,
    "final_answer": final_answer,
    }

def heuristic_score(
    answer,
    prompt=None,
    brevity_bonus=500.0,
    boxed_bonus=2.0,
    extract_bonus=1.0,
    fulltext_bonus=0.0,
):
    score = 0.0
    cand = extract_final_candidate(answer, fallback="none")
    if cand:
        score += boxed_bonus
    else:
        cand = extract_final_candidate(answer, fallback="number_only")
        if cand:
            score += extract_bonus
        else:
            cand = extract_final_candidate(
                answer, fallback="number_then_full"
            )
            if cand:
                score += fulltext_bonus
    score += 1.5 * math.exp(-len(answer) / brevity_bonus)
    return score


# Calculating next-token probabilities
@torch.inference_mode()
def calc_next_token_probas(model, tokenizer, prompt, device, show=True):
    token_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    logits = model(token_ids).squeeze(0)
    all_probas = torch.softmax(logits, dim=-1)

    t_idx = torch.arange(0, token_ids.shape[0] - 1, device=device)
    next_ids = token_ids[1:]
    next_token_probas = all_probas[t_idx, next_ids]
    prod_next_token_probas = torch.prod(next_token_probas)
    if show:
        print("Next-token probabilities:", next_token_probas)
        print("Joint probability:", prod_next_token_probas)
    else:
        return next_token_probas, prod_next_token_probas

# Calculating next-token log probabilities
@torch.inference_mode()
def calc_next_token_logprobas(model, tokenizer, prompt, device, show=True):

    token_ids = torch.tensor(tokenizer.encode(prompt), device=device)
    
    logits = model(token_ids.unsqueeze(0)).squeeze(0)
    all_logprobas = torch.log_softmax(logits, dim=-1)

    t_idx = torch.arange(0, token_ids.shape[0] - 1, device=device)
    next_ids = token_ids[1:]
    next_token_logprobas = all_logprobas[t_idx, next_ids]

    sum_next_token_logprobas = torch.sum(next_token_logprobas)
    
    if show:
        print("Next-token log probabilities:", next_token_logprobas)
        print("Joint log probability:", sum_next_token_logprobas)
    else:
        return next_token_logprobas, sum_next_token_logprobas

# Average log-probability scoring for answer tokens
@torch.inference_mode()
def avg_logprob_answer(model, tokenizer, prompt, answer, device="cpu"):
 
    prompt_ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(answer)
    full_ids = torch.tensor(prompt_ids + answer_ids, device=device)
 
    logits = model(full_ids.unsqueeze(0)).squeeze(0)
    logprobs = torch.log_softmax(logits, dim=-1)
 
    start = len(prompt_ids) - 1
    end = full_ids.shape[0] - 1
 
    t_idx = torch.arange(start, end, device=device)
    next_tokens = full_ids[start + 1 : end + 1]
    next_token_logps = logprobs[t_idx, next_tokens]
 
    return torch.mean(next_token_logps)

# Answer refinement
def make_refine_prompt(raw_prompt, draft, critique):
    return (
        "Revise the answer using the critique. Keep it concise and "
        "end with a final boxed result: \\boxed{ANSWER}\n\n"
        f"Question:\n{raw_prompt}\n\n"
        f"Previous answer:\n{draft}\n\n"
        f"Critique:\n{critique}\n\n"
        "Revised answer:"
    )
# Critique prompt and refinement plan
def make_critique_prompt(raw_prompt, draft):
    return (
        "You are a meticulous reviewer. Identify logical errors, missing "
        "steps, or arithmetic mistakes. If the answer seems correct, "
        "say so briefly. Then propose a concise plan to fix issues.\n\n"
        f"Question:\n{raw_prompt}\n\n"
        f"Draft answer:\n{draft}\n\n"
        "Write a short critique and bullet-point fix plan "
        "(under ~120 words).\n"
        "Critique:"
    )

# Self-refinement with multiple iterations and scoring support
def self_refinement_loop(
    model,
    tokenizer,
    raw_prompt,
    device,
    iterations=2,
    max_response_tokens=2048,
    max_critique_tokens=256,
    score_fn=None,
    prompt_renderer=render_template,
    prompt_suffix="",
    verbose=False,
    temperature=0.7,
    top_p=0.9,
):
    steps = []
    prompt = prompt_renderer(raw_prompt) + prompt_suffix
    current_full = generate_text_stream_concat_flex(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=max_response_tokens,
        verbose=False,
        generate_func=generate_text_top_p_stream_cache,
        temperature=temperature,
        top_p=top_p,
        )
    current_extracted = extract_final_candidate(
    current_full, fallback="number_then_full"
        )
    if score_fn:
        current_score = score_fn(answer=current_full, prompt=prompt)
    else:
        current_score = 0.0
    for it in range(iterations):
        draft_before_full = current_full
        draft_before_extracted = current_extracted
        score_before = current_score
        
        critique_prompt = make_critique_prompt(
            raw_prompt, draft_before_full
        )
        critique_full = generate_text_stream_concat_flex(
            model=model,
            tokenizer=tokenizer,
            prompt=critique_prompt,
            device=device,
            max_new_tokens=max_critique_tokens,
            verbose=False,
            generate_func=generate_text_top_p_stream_cache,
            temperature=temperature,
            top_p=top_p,
            )
        refine_prompt = make_refine_prompt(
            raw_prompt, draft_before_full, critique_full
            )
        revised_full = generate_text_stream_concat_flex(
            model=model,
            tokenizer=tokenizer,
            prompt=refine_prompt,
            device=device,
            max_new_tokens=max_response_tokens,
            verbose=False,
            generate_func=generate_text_top_p_stream_cache,
            temperature=temperature,
            top_p=top_p,
            )
        revised_extracted = extract_final_candidate(
            revised_full, fallback="number_then_full"
            )
        if score_fn:
            revised_score = score_fn(
            answer=revised_full, prompt=prompt
            )
        else:
            revised_score = 0.0

        step = {
            "iteration": it + 1,
            "draft_full": draft_before_full,
            "draft_extracted": draft_before_extracted,
            "critique": critique_full,
            "revised_full": revised_full,
            "revised_extracted": revised_extracted,
            "score_before": score_before,
            "score_after": revised_score,
            }
        steps.append(step)

        if verbose:
            print(
                f"[Refinement {it+1}/{iterations}]"
                f"\nCurrent: {draft_before_extracted}"
                f"\nRevised: {revised_extracted}"
                f"\nScore before: {score_before:.3f}"
                f"\nScore after: {revised_score:.3f}"
                f"\n{'=' * 25}\n"
            )
        if revised_score >= current_score:
            current_full = revised_full
            current_extracted = revised_extracted
            current_score = revised_score
    return {
        "final_full": current_full,
        "final_extracted": current_extracted,
        "steps": steps,
    }
