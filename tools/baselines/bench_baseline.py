#!/usr/bin/env python3
"""Decode-throughput baseline bench for the b12x KV/indexer dtype matrix.

Measures single-stream decode t/s at several context lengths against a running
vllm server (localhost:8000). Context is built by padding the prompt with
filler text tokenized server-side (/tokenize) so the target context token count
is accurate to ~1%.

Usage:
  python3 tools/baselines/bench_baseline.py --config fp8-fp8 \
      --contexts 100,8192,32768 --max-tokens 256 --repeats 2 \
      --out runs/baselines/decode_baselines.tsv

Appends TSV rows:
  date config context_target context_actual completion_tokens ttft_s decode_s decode_tps overall_tps
"""
import argparse, json, os, sys, time, urllib.request

URL = "http://localhost:8000"
MODEL = "Sebesky/MiniMax-M3-W4A16-GPTQ"

FILLER = (
    "The four-stroke internal combustion engine converts chemical energy into "
    "mechanical work through a repeating cycle of intake, compression, power, "
    "and exhaust strokes, with valve timing, ignition advance, and fuel "
    "metering all tuned to the engine's operating point. "
)
QUESTION = (
    "Summarize the text above in one sentence, then state how many strokes a "
    "four-stroke engine cycle has."
)


def _post(path, body, timeout=1200):
    req = urllib.request.Request(
        f"{URL}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def count_tokens(text):
    with _post("/tokenize", {"model": MODEL, "prompt": text}, timeout=120) as r:
        return int(json.load(r)["count"])


def build_prompt(target_tokens):
    """Filler repeated to ~target_tokens (chat overhead ~+50 tok is fine)."""
    if target_tokens <= 200:
        return "Explain how a four-stroke engine works."
    per = count_tokens(FILLER)
    reps = max((target_tokens - 100) // per, 1)
    text = FILLER * reps + "\n\n" + QUESTION
    # trim/extend to within 2%
    actual = count_tokens(text)
    while actual > target_tokens * 1.02 and reps > 1:
        reps -= max(int((actual - target_tokens) / per), 1)
        text = FILLER * max(reps, 1) + "\n\n" + QUESTION
        actual = count_tokens(text)
    return text


def run_once(prompt, max_tokens):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
        # keep decode pure: no thinking budget shenanigans; template default
    }
    t0 = time.time(); t_first = None; usage = None; text_head = []
    with _post("/v1/chat/completions", body) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            ch = obj.get("choices") or []
            if ch:
                delta = ch[0].get("delta", {})
                piece = delta.get("content") or delta.get("reasoning")
                if piece:
                    if t_first is None:
                        t_first = time.time()
                    if len(text_head) < 20:
                        text_head.append(piece)
    t_end = time.time()
    if t_first is None or usage is None:
        raise RuntimeError("no streamed tokens / no usage returned")
    ct = usage["completion_tokens"]
    pt = usage["prompt_tokens"]
    ttft = t_first - t0
    decode_s = t_end - t_first
    return dict(prompt_tokens=pt, completion_tokens=ct, ttft=ttft,
                decode_s=decode_s, decode_tps=ct / decode_s if decode_s > 0 else float("nan"),
                overall_tps=ct / (t_end - t0), head="".join(text_head).replace("\n", " ")[:100])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="label, e.g. bf16-fp8 / fp8-fp8 / nvfp4-fp8 / nvfp4-nvfp4")
    ap.add_argument("--contexts", default="100,8192,32768", help="target prompt token counts")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", default="runs/baselines/decode_baselines.tsv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    new_file = not os.path.exists(args.out)
    date = time.strftime("%Y-%m-%d %H:%M")
    with open(args.out, "a") as f:
        if new_file:
            f.write("date\tconfig\tctx_target\tprompt_tokens\tcompletion_tokens\tttft_s\tdecode_s\tdecode_tps\toverall_tps\n")
        for ctx in [int(x) for x in args.contexts.split(",")]:
            prompt = build_prompt(ctx)
            # warm the prefix cache / graphs once, unmeasured
            run_once(prompt, 16)
            for rep in range(args.repeats):
                m = run_once(prompt, args.max_tokens)
                row = (f"{date}\t{args.config}\t{ctx}\t{m['prompt_tokens']}\t{m['completion_tokens']}"
                       f"\t{m['ttft']:.3f}\t{m['decode_s']:.3f}\t{m['decode_tps']:.2f}\t{m['overall_tps']:.2f}")
                f.write(row + "\n"); f.flush()
                print(f"[{args.config} ctx={ctx} rep={rep}] prompt={m['prompt_tokens']} "
                      f"gen={m['completion_tokens']} ttft={m['ttft']:.2f}s "
                      f"decode={m['decode_tps']:.2f} t/s | {m['head']!r}", flush=True)


if __name__ == "__main__":
    main()
