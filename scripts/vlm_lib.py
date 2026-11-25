"""Model helpers: load Qwen3-VL, build prompts, and score class labels.

Scoring. The model never writes free text here. For each sample we read the
next-token logits right after the prompt and compare only the first tokens of
the allowed answers (e.g. "Grab", "Place", ... or "Yes"/"No"). A softmax over
those few logits gives a probability for every class. This:
  - removes answer-parsing errors (no "I think it is grabbing..." to parse),
  - gives calibrated-looking scores needed for conformal prediction sets,
  - is identical for zero-shot, text-only and fine-tuned models.
We check at start-up that every class name starts with a different token.
"""
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

SKILL_PROMPT = ("The image shows {n} frames from a robot's head camera, in time order "
                "(left to right, then top to bottom), taken during one short task. "
                "Which basic skill is the robot performing? Answer with one word from this list: {classes}.")
SKILL_PROMPT_BLIND = ("A robot is performing one short task. Which basic skill is it performing? "
                      "Answer with one word from this list: {classes}.")
SUCCESS_PROMPT = ("The image shows {n} frames from the end of a robot episode, in time order "
                  "(left to right, then top to bottom). The robot was asked: \"{instruction}\" "
                  "Did the robot succeed? Answer Yes or No.")
SUCCESS_PROMPT_BLIND = ("A robot was asked: \"{instruction}\" Did the robot succeed? Answer Yes or No.")
SUCCESS_CLASSES = ["Yes", "No"]  # index 0 = success


def load_model(name: str, quant: str = "bf16", adapter: str | Path | None = None):
    qconf = None
    if quant == "nf4":
        qconf = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForImageTextToText.from_pretrained(
        name, dtype=torch.bfloat16, quantization_config=qconf, device_map="cuda", attn_implementation="sdpa")
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    processor = AutoProcessor.from_pretrained(name)
    return model, processor


def messages(sample: dict, classes: list[str], cfg: dict, with_image: bool, answer: str | None = None) -> list:
    n = cfg["frames_per_clip"]
    if sample["task"] == "skill":
        text = (SKILL_PROMPT if with_image else SKILL_PROMPT_BLIND).format(n=n, classes=", ".join(classes))
    else:
        text = (SUCCESS_PROMPT if with_image else SUCCESS_PROMPT_BLIND).format(n=n, instruction=sample["instruction"])
    content = ([{"type": "image", "image": sample["image_path"]}] if with_image else []) + [{"type": "text", "text": text}]
    msgs = [{"role": "user", "content": content}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return msgs


def encode(processor, msgs: list, add_generation_prompt: bool):
    return processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=add_generation_prompt,
                                         return_dict=True, return_tensors="pt")


def first_token_ids(processor, classes: list[str]) -> list[int]:
    ids = [processor.tokenizer.encode(c, add_special_tokens=False)[0] for c in classes]
    assert len(set(ids)) == len(ids), f"class names share a first token: {classes}"
    return ids


@torch.no_grad()
def score(model, processor, msgs: list, class_ids: list[int]) -> tuple[list[float], float]:
    """Return (probabilities over classes, latency in ms) for one sample (batch size 1)."""
    enc = encode(processor, msgs, add_generation_prompt=True).to(model.device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = model(**enc, logits_to_keep=1).logits[0, -1].float()
    torch.cuda.synchronize()
    ms = 1000 * (time.perf_counter() - t0)
    probs = torch.softmax(logits[class_ids], dim=0).cpu().tolist()
    return probs, ms


def skill_classes(cfg: dict) -> list[str]:
    path = cfg["run_dir"] / "classes.json"
    return json.loads(path.read_text())


def save_skill_classes(cfg: dict, classes: list[str]) -> None:
    (cfg["run_dir"] / "classes.json").write_text(json.dumps(classes))

