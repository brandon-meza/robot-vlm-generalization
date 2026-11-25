"""Step 4: LoRA fine-tune of Qwen3-VL on one split's training part.

Usage:
  python scripts/04_finetune_lora.py configs/X.json --split random_episode
  python scripts/04_finetune_lora.py configs/X.json --split random_episode --shuffle-labels

What is trained. Small low-rank adapters (LoRA, rank r) are added to every
linear layer of the language model. The vision encoder and all original
weights stay frozen. Only the answer tokens contribute to the loss (the
prompt and image tokens are masked with -100), so the model learns to answer,
not to copy the prompt.

--shuffle-labels is a control: labels are permuted at random inside each task
before training. A model trained on shuffled labels can only learn the label
frequencies. If it scores close to the real fine-tune, the real fine-tune did
not learn anything from the images either.

Memory: gradient checkpointing is on and model.train() is called (without
train mode, checkpointing is silently disabled and memory roughly doubles).
"""
import argparse
import json
import math
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model

from common import load_config, load_split
from vlm_lib import encode, load_model, messages, save_skill_classes, skill_classes


def build_batch(processor, rows: list[dict], classes: list[str], cfg: dict):
    """Tokenise prompt+answer, mask everything except the answer tokens."""
    ids, labels, pix, grids, mmt = [], [], [], [], []
    for r in rows:
        full = encode(processor, messages(r, classes, cfg, True, answer=r["train_label"]), False)
        prompt = encode(processor, messages(r, classes, cfg, True), True)
        n_prompt = prompt["input_ids"].shape[1]
        x = full["input_ids"][0]
        y = x.clone()
        y[:n_prompt] = -100
        ids.append(x)
        labels.append(y)
        pix.append(full["pixel_values"])
        grids.append(full["image_grid_thw"])
        mmt.append(full["mm_token_type_ids"][0])  # marks image tokens, needed for multimodal RoPE
    L = max(len(x) for x in ids)
    pad = processor.tokenizer.pad_token_id
    input_ids = torch.full((len(ids), L), pad, dtype=torch.long)
    lab = torch.full((len(ids), L), -100, dtype=torch.long)
    att = torch.zeros((len(ids), L), dtype=torch.long)
    mm = torch.zeros((len(ids), L), dtype=torch.long)
    for i, (x, y, t) in enumerate(zip(ids, labels, mmt)):  # right padding for training
        input_ids[i, :len(x)] = x
        lab[i, :len(y)] = y
        att[i, :len(x)] = 1
        mm[i, :len(t)] = t
    return {"input_ids": input_ids, "attention_mask": att, "labels": lab, "mm_token_type_ids": mm,
            "pixel_values": torch.cat(pix), "image_grid_thw": torch.cat(grids)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--split", required=True)
    ap.add_argument("--shuffle-labels", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    data = load_split(cfg, args.split)
    if not (cfg["run_dir"] / "classes.json").exists():
        from common import load_episodes
        save_skill_classes(cfg, load_episodes(cfg)["skill"].value_counts().index.tolist())
    classes = skill_classes(cfg)
    train = data[data.part == "train"].reset_index(drop=True)
    train["train_label"] = train["label"]
    if args.shuffle_labels:
        for task in ("skill", "success"):
            m = train.task == task
            train.loc[m, "train_label"] = rng.permutation(train.loc[m, "label"].to_numpy())
    tag = f"{args.split}__{'shuffled' if args.shuffle_labels else 'finetuned'}"
    out = cfg["run_dir"] / "adapters" / tag

    model, processor = load_model(cfg["model"], "bf16")
    lora = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                      # language-model projections only; the vision tower stays frozen
                      target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()

    bs, accum = cfg["batch_size"], cfg["grad_accum"]
    steps_per_epoch = math.ceil(len(train) / (bs * accum))
    total = cfg["max_steps"] or steps_per_epoch * cfg["epochs"]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"], weight_decay=0.0)
    warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)))
    log, step, t0 = [], 0, time.time()
    order = []
    while step < total:
        if not order:
            order = list(rng.permutation(len(train)))
        loss_acc = 0.0
        for _ in range(accum):
            idx, order = order[:bs], order[bs:]
            if not idx:
                order = list(rng.permutation(len(train)))
                idx, order = order[:bs], order[bs:]
            batch = build_batch(processor, [train.iloc[i].to_dict() for i in idx], classes, cfg)
            batch = {k: v.to("cuda") for k, v in batch.items()}
            loss = model(**batch).loss / accum
            loss.backward()
            loss_acc += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        log.append({"step": step, "loss": loss_acc, "lr": sched.get_last_lr()[0], "elapsed_s": time.time() - t0})
        if step % 10 == 0 or step == total:
            print(f"{tag} step {step}/{total} loss {loss_acc:.4f} "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB {time.time() - t0:.0f}s", flush=True)
    model.save_pretrained(out)
    (out / "train_log.json").write_text(json.dumps({"tag": tag, "n_train": len(train), "steps": total,
                                                    "batch_size": bs, "grad_accum": accum,
                                                    "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                                                    "log": log}, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
