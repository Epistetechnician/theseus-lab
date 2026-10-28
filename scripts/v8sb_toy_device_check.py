"""Toy end-to-end check that the v8sb GPU shim matches the CPU path (run from the repo root)."""
import json, os, sys, tempfile
from pathlib import Path
from unittest import mock
import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM, OPTConfig, OPTForCausalLM, PreTrainedTokenizerFast
import replacement_loss_v3_runner as v3
import replacement_loss_v8sb_runner as r


def build(root):
    tdir = root / "tok"; tdir.mkdir()
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}; vocab.update({f"token_{i}": i + 4 for i in range(60)})
    b = Tokenizer(WordLevel(vocab, unk_token="<unk>")); b.pre_tokenizer = Whitespace()
    b.post_processor = TemplateProcessing(single="<bos> $A", pair="<bos> $A $B", special_tokens=[("<bos>", 1)])
    PreTrainedTokenizerFast(tokenizer_object=b, pad_token="<pad>", bos_token="<bos>", eos_token="<eos>", unk_token="<unk>", model_max_length=64).save_pretrained(tdir)
    torch.manual_seed(0)
    specs = (("opt_relu", OPTForCausalLM(OPTConfig(vocab_size=64, hidden_size=16, ffn_dim=32, num_hidden_layers=2, num_attention_heads=4, word_embed_proj_dim=16, max_position_embeddings=64, bos_token_id=1, eos_token_id=2, pad_token_id=0))),
             ("gpt2_gelu", GPT2LMHeadModel(GPT2Config(vocab_size=64, n_embd=16, n_layer=2, n_head=4, n_positions=64, n_ctx=64, bos_token_id=1, eos_token_id=2, pad_token_id=0))),
             ("llama_silu_gated", LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64, bos_token_id=1, eos_token_id=2, pad_token_id=0))))
    cps = []
    for fam, m in specs:
        p = root / "ckpt" / fam; p.mkdir(parents=True); m.save_pretrained(p, safe_serialization=True)
        for a in tdir.iterdir(): (p / a.name).write_bytes(a.read_bytes())
        cps.append(v3.Checkpoint(fam, f"toy-{fam}", "a" * 40, p, True))
    return cps


def run(device, cps_src):
    os.environ[r.DEVICE_ENV] = device
    with tempfile.TemporaryDirectory() as t:
        root = Path(t).resolve()
        import shutil
        cps = []
        for c in cps_src:
            dst = root / "ckpt" / c.family; shutil.copytree(c.local_path, dst)
            cps.append(v3.Checkpoint(c.family, c.model_id, c.revision, dst, True))
        manifest = root / "m.json"; manifest.write_text(json.dumps({"protocol_id": "toy"}))
        import secrets, itertools, hashlib
        counter = itertools.count()
        det = lambda n=32: hashlib.sha256(str(next(counter)).encode()).digest()[:n]
        with mock.patch.object(v3.os, "urandom", det), mock.patch.object(secrets, "token_bytes", det), mock.patch.object(v3, "verify_packet", return_value={"packet_sha256": "d" * 64}), mock.patch.object(v3, "_load_checkpoints", return_value=cps):
            with r._device_shim():
                v3.prepare_candidates(root, manifest); v3.freeze_candidates(root); v3.make_challenges(root, manifest); v3.evaluate_packet(root, manifest)
        rep = json.loads((root / "replacement-loss-v3-report.json").read_text())
        out = []
        for c in rep["cases"]:
            out.append((c["family"], c["seed_index"], c["case_status"], c["primary_effects"]["treatment"], c["mapping_aligned_permutation_diagnostic"]))
        return out


with tempfile.TemporaryDirectory() as t:
    cps = build(Path(t).resolve())
    cpu = run("cpu", cps); gpu = run("cuda", cps)
    worst = 0.0
    for a, b in zip(cpu, gpu):
        assert a[:2] == b[:2]
        for k in a[3]:
            x, y = a[3][k], b[3][k]
            if isinstance(x, dict):
                for kk in x:
                    if isinstance(x[kk], dict):
                        for k3 in x[kk]:
                            if isinstance(x[kk][k3], (int, float)) and isinstance(y[kk].get(k3), (int, float)):
                                d = abs(x[kk][k3] - y[kk][k3]); worst = max(worst, d)
                                if d > 0.02: print("  diff", a[0], a[1], k, kk, k3, x[kk][k3], y[kk][k3])
                    elif isinstance(x[kk], (int, float)) and isinstance(y[kk], (int, float)):
                        d = abs(x[kk] - y[kk]); worst = max(worst, d)
                        if d > 0.02: print("  diff", a[0], a[1], k, kk, x[kk], y[kk])
            elif isinstance(x, (int, float)): worst = max(worst, abs(x - y))
        print(a[0], a[1], a[2], b[2], "equiv", b[4]["module_effect_max_abs_error_vs_treatment"], b[4]["logit_effect_max_abs_error_vs_treatment"])
    print("cases", len(cpu), len(gpu), "max |cpu-gpu| primary treatment effect metric diff", worst)
