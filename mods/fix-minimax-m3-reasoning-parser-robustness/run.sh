#!/bin/bash
set -euo pipefail

# MiniMax-M3 reasoning-parser robustness fix (v3).
#
# Three independent bugs, three fixes:
#
# 1) BACKPORT upstream d8e422ccd "[Bugfix] Parse MiniMax M3 streaming
#    reasoning by text markers (#45718)" (merged after this image's vLLM).
#    The image's parser detects the <mm:think>/</mm:think> markers by SINGLE
#    vocabulary token id. The markers exist as single vocab entries, but the
#    model can (and at temp>0 does) emit the marker text spelled out in
#    smaller ordinary tokens. The id check then never fires: the stream never
#    flips to content, and the literal "</mm:think>" text leaks into the
#    visible output. Downstream, the chat template ("all-turn visible
#    thinking") splits assistant history on that leaked tag, so the polluted
#    turn re-enters the context garbled -- the model then blurs the
#    reasoning/answer boundary in later turns (over-reasoning, stray/nested
#    think markers). The backported parser matches marker TEXT with
#    partial-marker buffering at chunk boundaries, fixing streaming
#    extraction, is_reasoning_end_streaming (tool-call gating), and
#    extract_content_ids.
#
#    We replace the whole file with the upstream version
#    (minimax_m3_reasoning_parser_45718.py, pristine copy of d8e422ccd)
#    instead of sed-patching: the diff touches most of the file.
#
# 2) RE-APPLY the non-streaming dropped-closer recovery on top. When the
#    model genuinely never emits the closer (sampling variance on post-tool
#    turns), upstream still routes the ENTIRE answer into reasoning_content
#    and returns content=None; clients that read `content` lose the visible
#    answer (tool-eval-bench Structured Output failures TC-65/66/67/69).
#    Recovery extracts a trailing JSON object/array or markdown fence as
#    content, else returns the text as content. #45718 does not change
#    extract_reasoning, so the anchors apply cleanly to the backport.
#
# 3) FIX is_reasoning_end prompt poisoning (thinking_mode=adaptive leak).
#    The M3 chat template mentions BOTH markers verbatim in its
#    <thinking_instructions> system text ("wrap your reasoning in
#    <mm:think></mm:think> tags..."), and that text tokenizes to the real
#    single-vocab special ids. is_reasoning_end(prompt_token_ids) -- used to
#    seed the streaming state (parser/abstract_parser.py parse_delta,
#    chat_completion/serving.py, structured_output) -- did a whole-prompt
#    "last marker wins" scan, so every adaptive prompt (no generation
#    prefill) reports "reasoning already ended": the parser is bypassed and
#    the model's reasoning plus a literal <mm:think> leak into content.
#    thinking_mode=enabled only survived by luck (its <mm:think> generation
#    prefill is the last prompt token, outranking the instruction text).
#    Fix: only a marker prefilled at the very END of the prompt seeds the
#    state; ...<mm:think> -> in reasoning, ...</mm:think> (disabled mode
#    prefill) -> content-only, anything else (adaptive) -> not ended, the
#    text-marker streaming parser resolves what the model chooses to do.
#
# REMOVED from v1: the streaming fence/JSON "answer boundary" heuristic
# (_stream_answer_boundary). It existed only to mask bug (1) mid-stream and
# misfired on the first ``` fence or JSON-ish text INSIDE legitimate
# reasoning, permanently flipping the stream to content -- reasoning then
# rendered as the final answer, and the polluted history caused the same
# boundary-blurring feedback loop. With text-marker parsing there is nothing
# for it to recover; a genuinely dropped closer mid-stream is unrecoverable
# in-flight and is left as reasoning (the non-streaming path still recovers
# it at finish).

PYTHON=${PYTHON:-python3}
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MOD_DIR
$PYTHON - <<'PY'
import os
from pathlib import Path

path = Path('/usr/local/lib/python3.12/dist-packages/vllm/reasoning/minimax_m3_reasoning_parser.py')
backport = Path(os.environ['MOD_DIR']) / 'minimax_m3_reasoning_parser_45718.py'

text = path.read_text()
if (
    'def _recover_unclosed_reasoning' in text
    and '_strip_partial_marker_suffix' in text
    and 'prefilled at the very END of the prompt' in text
):
    print('MiniMax-M3 reasoning-parser robustness fix (v3) already applied:', path)
    raise SystemExit(0)

# ---- 1) wholesale backport of upstream d8e422ccd (#45718) ----
text = backport.read_text()
assert '_strip_partial_marker_suffix' in text, 'backport file is not the #45718 version'
assert 'def _recover_unclosed_reasoning' not in text, 'backport file unexpectedly pre-patched'

# ---- 2) non-streaming dropped-closer recovery (unchanged by #45718) ----

# 2a) _initial_in_reasoning branch with a dropped closer.
old_b1 = (
    '        if self._initial_in_reasoning and self.start_token not in model_output:\n'
    '            reasoning, end, content = model_output.partition(self.end_token)\n'
    '            if not end:\n'
    '                return model_output, None\n'
    '            return reasoning, content or None\n'
)
new_b1 = (
    '        if self._initial_in_reasoning and self.start_token not in model_output:\n'
    '            reasoning, end, content = model_output.partition(self.end_token)\n'
    '            if not end:\n'
    '                # Closer dropped: recover the answer (never empty content).\n'
    '                return self._recover_unclosed_reasoning(model_output)\n'
    '            return reasoning, content or None\n'
)
assert text.count(old_b1) == 1, 'anchor 2a (non-streaming branch 1) not found'
text = text.replace(old_b1, new_b1, 1)

# 2b) explicit <mm:think> present but closer dropped.
old_b2 = (
    '        content_before, _, after_start = model_output.partition(self.start_token)\n'
    '        reasoning, end, content_after = after_start.partition(self.end_token)\n'
    '        if not end:\n'
    '            return reasoning, content_before or None\n'
)
new_b2 = (
    '        content_before, _, after_start = model_output.partition(self.start_token)\n'
    '        reasoning, end, content_after = after_start.partition(self.end_token)\n'
    '        if not end:\n'
    '            rec_reasoning, rec_content = self._recover_unclosed_reasoning(reasoning)\n'
    '            return rec_reasoning, (content_before + (rec_content or "")) or None\n'
)
assert text.count(old_b2) == 1, 'anchor 2b (non-streaming branch 2) not found'
text = text.replace(old_b2, new_b2, 1)

# 2c) recovery helper (non-streaming only; no streaming heuristics).
helpers = (
    '    def _recover_unclosed_reasoning(self, text):\n'
    '        # Model emitted a reasoning block but dropped the closing\n'
    '        # </mm:think> token. Without this, the entire answer is classified\n'
    '        # as reasoning and content is empty. Recover the answer instead.\n'
    '        import json as _json\n'
    '        import re as _re\n'
    '        decoder = _json.JSONDecoder()\n'
    '        for _m in _re.finditer(r"[\\{\\[]", text):\n'
    '            i = _m.start()\n'
    '            try:\n'
    '                _obj, _end = decoder.raw_decode(text[i:])\n'
    '            except ValueError:\n'
    '                continue\n'
    '            reasoning = text[:i].strip() or None\n'
    '            return reasoning, text[i : i + _end].strip() or None\n'
    '        fence = text.find("```")\n'
    '        if fence != -1:\n'
    '            reasoning = text[:fence].strip() or None\n'
    '            content = text[fence:].strip() or None\n'
    '            if content:\n'
    '                return reasoning, content\n'
    '        # Never bury a real answer in reasoning with empty content.\n'
    '        return None, text\n'
    '\n'
    '    def is_reasoning_end_streaming(\n'
)
anchor = '    def is_reasoning_end_streaming(\n'
assert text.count(anchor) == 1, 'anchor 2c (is_reasoning_end_streaming) not found'
text = text.replace(anchor, helpers, 1)

# ---- 3) is_reasoning_end: only an end-of-prompt prefill marker counts ----
old_b3 = (
    '    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:\n'
    '        start_index = self._rfind_token_sequence(input_ids, self._start_token_ids)\n'
    '        end_index = self._rfind_token_sequence(input_ids, self._end_token_ids)\n'
    '        if end_index < 0:\n'
    '            return False\n'
    '        if start_index < 0:\n'
    '            return True\n'
    '        return end_index > start_index\n'
)
new_b3 = (
    '    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:\n'
    '        # Called with PROMPT token ids to seed the streaming state. Only a\n'
    '        # marker prefilled at the very END of the prompt is meaningful: the\n'
    '        # M3 chat template mentions both markers verbatim in its\n'
    '        # <thinking_instructions> text (tokenizing to the real special\n'
    '        # ids), so a whole-prompt scan reports "reasoning ended" for every\n'
    '        # thinking_mode=adaptive request and the parser is bypassed\n'
    '        # (reasoning + literal <mm:think> leak into content). Tails:\n'
    '        #   ...<mm:think>   enabled prefill  -> inside reasoning -> False\n'
    '        #   ...</mm:think>  disabled prefill -> content only     -> True\n'
    '        #   anything else   adaptive         -> False; the text-marker\n'
    '        #                   streaming parser handles whether the model\n'
    '        #                   opens a block or answers directly.\n'
    '        end_len = len(self._end_token_ids)\n'
    '        return bool(\n'
    '            end_len\n'
    '            and len(input_ids) >= end_len\n'
    '            and tuple(input_ids[-end_len:]) == tuple(self._end_token_ids)\n'
    '        )\n'
)
assert text.count(old_b3) == 1, 'anchor 3 (is_reasoning_end) not found'
text = text.replace(old_b3, new_b3, 1)

path.write_text(text)

import py_compile
py_compile.compile(str(path), doraise=True)
print('Applied MiniMax-M3 reasoning parser v3 (#45718 backport + non-streaming recovery + adaptive prompt-poisoning fix):', path)
PY
