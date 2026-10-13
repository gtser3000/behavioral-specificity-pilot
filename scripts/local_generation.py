"""Shared offline inference helpers for timing and an explicitly approved test."""
import os
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def load_local_pair():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This setup requires a CUDA GPU supporting BF16.')
    torch.set_num_threads(4)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(ROOT / '.cache/model', local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    base = AutoModelForCausalLM.from_pretrained(ROOT / '.cache/model', local_files_only=True,
        use_safetensors=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    model = PeftModel.from_pretrained(base, ROOT / 'runs/medical-full-epoch/adapter', is_trainable=False)
    model.config.use_cache = True
    model.requires_grad_(False)
    model.eval()
    torch.cuda.synchronize()
    return model, tokenizer, time.perf_counter() - started


def generate(model, tokenizer, prompts, condition, max_new_tokens, seed=0,
             fixed_length=False, do_sample=False, return_text=False):
    if condition not in {'original', 'fine_tuned'}:
        raise ValueError('Unknown model condition')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rendered = [tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                tokenize=False, add_generation_prompt=True) for prompt in prompts]
    encoded = tokenizer(rendered, padding=True, return_tensors='pt').to('cuda')
    eos_ids = model.generation_config.eos_token_id
    generation = GenerationConfig(
        max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens if fixed_length else 0,
        do_sample=do_sample, temperature=1.0 if do_sample else None,
        top_p=1.0 if do_sample else None, top_k=0 if do_sample else None,
        repetition_penalty=1.0, num_beams=1, use_cache=True,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=eos_ids,
        bos_token_id=model.generation_config.bos_token_id,
    )
    context = model.disable_adapter() if condition == 'original' else nullcontext()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with context, torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        output = model.generate(**encoded, generation_config=generation)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    continuations = output[:, encoded['input_ids'].shape[1]:].cpu().tolist()
    eos_set = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
    lengths = []
    truncated = []
    clean_tokens = []
    for sequence in continuations:
        end = next((i + 1 for i, token in enumerate(sequence) if token in eos_set), len(sequence))
        actual = sequence[:end]
        # For forced-length timing, all steps were computed even if an EOS
        # would have been produced in an unconstrained response.
        lengths.append(len(sequence) if fixed_length else len(actual))
        truncated.append(not fixed_length and len(sequence) >= max_new_tokens and not any(t in eos_set for t in sequence))
        clean_tokens.append(actual)
    measured = {'condition': condition, 'batch_size': len(prompts), 'seconds': elapsed,
        'generated_tokens': sum(lengths), 'tokens_per_second': sum(lengths) / elapsed,
        'token_counts': lengths, 'truncated': truncated,
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
        'peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024**3,
        'seed': seed, 'max_new_tokens': max_new_tokens, 'fixed_length': fixed_length,
        'do_sample': do_sample}
    if return_text:
        measured['responses'] = tokenizer.batch_decode(clean_tokens, skip_special_tokens=True)
    return measured
