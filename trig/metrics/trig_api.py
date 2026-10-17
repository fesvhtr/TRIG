"""TRIGScore via any OpenAI-compatible chat endpoint (vLLM, OpenAI, DashScope, ...).

The judge is asked for one of five rating words; the score is the probability-weighted
expectation over those words computed from the first generated token's top-logprobs.

Notes for recent judges (Qwen3.5 / Qwen3-VL):
  * They run in *thinking* mode by default. Thinking must be disabled, otherwise the first
    generated token is ``<think>`` and no rating token is present in the logprobs.
    We pass ``chat_template_kwargs={"enable_thinking": False}`` through ``extra_body``.
  * Tokenization of the rating words differs between tokenizers (e.g. Qwen2.5 splits
    "terrible" as "terr|ible", Qwen3.5 as "ter|rible"). Tokens are therefore matched by
    prefix against the rating words instead of a hard-coded token list.
"""
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from tqdm import tqdm

from trig.metrics.base import BaseMetric
from trig.utils.utils import encode_image
from trig.config import gpt_logit_system_msg, gpt_logit_dimension_msg

# Ordered from worst to best; weight of word i is i / (n - 1): 0, .25, .5, .75, 1.
RATING_WORDS = ["terrible", "bad", "medium", "good", "excellent"]
RATING_WEIGHTS = {w: i / (len(RATING_WORDS) - 1) for i, w in enumerate(RATING_WORDS)}
_LEADING_SPACE_MARKS = ("Ġ", "▁", " ", "_")


def normalize_token(token: str) -> str:
    """Lower-case a token and strip BPE/SentencePiece leading-space markers and punctuation."""
    t = token
    while t and t[0] in _LEADING_SPACE_MARKS:
        t = t[1:]
    return "".join(ch for ch in t.lower() if ch.isalpha())


def token_to_rating_word(token: str):
    """Map a (possibly partial) token to one of RATING_WORDS, or None.

    A token matches a word when it is a prefix of the word (``ex`` -> excellent,
    ``ter`` -> terrible) or the word is a prefix of the token (``good.`` -> good).
    Single-letter tokens are ignored because ``b``/``t``/``e`` are ambiguous.
    """
    n = normalize_token(token)
    if len(n) < 2:
        return None
    matches = [w for w in RATING_WORDS if w.startswith(n) or n.startswith(w)]
    return matches[0] if len(matches) == 1 else None


def logprobs_to_score(top_logprobs, confidence: bool = False, return_details: bool = False):
    """Expected rating in [0, 1] from a list of objects with ``.token`` and ``.logprob``."""
    word_probs = {}
    for item in top_logprobs:
        word = token_to_rating_word(item.token)
        if word is None:
            continue
        try:
            p = math.exp(float(item.logprob))
        except OverflowError:
            p = 0.0
        word_probs[word] = word_probs.get(word, 0.0) + p

    if not word_probs:
        return (0.0, {}) if return_details else 0.0

    total = sum(word_probs.values()) + 1e-10
    normalized = {w: p / total for w, p in word_probs.items()}
    score = sum(RATING_WEIGHTS[w] * p for w, p in normalized.items())
    if confidence:
        score *= max(normalized.values())
    return (score, normalized) if return_details else score


class TRIGAPIMetric(BaseMetric):
    def __init__(self, API_KEY="EMPTY", endpoint="http://localhost:8000/v1/", model_name="None",
                 top_logprobs=10, temperature=1.0, max_tokens=8, enable_thinking=False,
                 num_workers=8, max_retries=3, confidence=False, **kwargs):
        super().__init__(**kwargs)
        # TRIG_JUDGE_ENDPOINT lets Slurm jobs point at whichever node serves the judge
        # without editing the yaml.
        endpoint = os.environ.get("TRIG_JUDGE_ENDPOINT", endpoint)
        API_KEY = os.environ.get("TRIG_JUDGE_API_KEY", API_KEY)
        print("Initializing TRIGAPIMetric: endpoint={}, model_name={}, dimension={}, "
              "top_logprobs={}, enable_thinking={}".format(endpoint, model_name, self.dimension,
                                                            top_logprobs, enable_thinking))
        self.top_logprobs = top_logprobs
        self.client = openai.Client(api_key=API_KEY, base_url=endpoint)
        self.task = None
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.num_workers = num_workers
        self.max_retries = max_retries
        self.confidence = confidence
        self.metric_name = "TRIGAPIMetric_" + model_name
        self._warned_think = False

    # ------------------------------------------------------------------ prompts
    def get_conv_template(self):
        name = self.model_name.lower()
        for key in ("qwen", "gpt", "llava"):
            if key in name:
                return key
        return "generic"

    def _extra_body(self):
        """Request options that only OpenAI-compatible local servers understand."""
        if "gpt" in self.model_name.lower():
            return None
        # Qwen3 / Qwen3.5 chat templates honour enable_thinking; other templates ignore it.
        return {"chat_template_kwargs": {"enable_thinking": bool(self.enable_thinking)}}

    def format_msg(self, conv_temp, prompt, gen_image, *args):
        image_part = {"type": "image_url",
                      "image_url": {"url": f"data:image/{gen_image['type']};base64,{gen_image['base64']}"}}
        if len(args) == 0:  # t2i
            system = gpt_logit_system_msg[self.task].format(gpt_logit_dimension_msg[self.task][self.dimension])
            text = 'Prompt for generating this image: ' + prompt
        elif len(args) == 1:  # p2p (source image is not shown to the judge, as in the paper)
            system = gpt_logit_system_msg[self.task].format(gpt_logit_dimension_msg[self.task][self.dimension])
            text = 'Editing prompt for generating this image: ' + prompt
        elif len(args) == 2:  # s2p
            item = args[1]
            system = gpt_logit_system_msg[self.task].format(item, gpt_logit_dimension_msg[self.task][self.dimension])
            text = 'Subject-driven Prompt for generating this image: ' + prompt
        else:
            raise ValueError("Unexpected number of extra arguments: {}".format(len(args)))
        return [{"role": "system", "content": system},
                {"role": "user", "content": [{"type": "text", "text": text}, image_part]}]

    # ------------------------------------------------------------------ scoring
    def compute(self, image_path, prompt, *args, return_details=False):
        gen_image = encode_image(image_path)
        src_img = encode_image(args[0]) if len(args) >= 1 else None
        item = args[1] if len(args) == 2 else None
        msg_args = [arg for arg in [gen_image, src_img, item] if arg is not None]
        msg = self.format_msg(self.get_conv_template(), prompt, *msg_args)

        request = dict(model=self.model_name, messages=msg, logprobs=True,
                       temperature=self.temperature, top_logprobs=self.top_logprobs,
                       max_tokens=self.max_tokens)
        extra_body = self._extra_body()
        if extra_body:
            request["extra_body"] = extra_body

        last_error = None
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(**request)
                content = completion.choices[0].logprobs.content
                if not content:
                    raise RuntimeError("empty logprobs content")
                first = content[0]
                if "<think>" in first.token and not self._warned_think:
                    self._warned_think = True
                    print("[TRIGAPIMetric] WARNING: judge answered with <think>; disable thinking "
                          "(enable_thinking=False) or the scores will be 0.")
                score, details = logprobs_to_score(first.top_logprobs, confidence=self.confidence,
                                                   return_details=True)
                score = round(score, 3)
                if return_details:
                    return score, {"first_token": first.token, "word_probs": details,
                                   "text": completion.choices[0].message.content}
                return score
            except Exception as e:  # network / server hiccups
                last_error = e
        print(f"[TRIGAPIMetric] Error after {self.max_retries} attempts on {image_path}: {last_error}")
        return (0.0, {"error": str(last_error)}) if return_details else 0.0

    def logprobs_score(self, top_logprobs, confidence=False):
        """Backward-compatible wrapper around :func:`logprobs_to_score`."""
        return logprobs_to_score(top_logprobs, confidence=confidence)

    def _compute_one(self, task, data):
        if task == 't2i':
            return self.compute(data['gen_image_path'], data['prompt'])
        if task == 'p2p':
            return self.compute(data['gen_image_path'], data['prompt'], data['img_id'])
        if task == 's2p':
            return self.compute(data['gen_image_path'], data['prompt'], data['img_id'], data['item'])
        raise ValueError(f"Unsupported task: {task}")

    def compute_batch(self, task, promp_data):
        if task is None:
            raise ValueError("task is None")
        self.task = task
        results = {}
        if self.num_workers <= 1:
            for data in tqdm(promp_data):
                results[data['data_id']] = self._compute_one(task, data)
            return results
        # vLLM batches concurrent requests, so a small thread pool gives a large speed-up.
        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futures = {pool.submit(self._compute_one, task, d): d['data_id'] for d in promp_data}
            for fut in tqdm(as_completed(futures), total=len(futures)):
                results[futures[fut]] = fut.result()
        return results


if __name__ == "__main__":
    # Smoke test against a running judge. Point TRIG_JUDGE_ENDPOINT at the vLLM server, e.g.
    #   TRIG_JUDGE_ENDPOINT=http://gpu-node:8000/v1/ python -m trig.metrics.trig_api
    model = os.environ.get("TRIG_JUDGE_MODEL", "Qwen/Qwen3.5-9B")
    metric = TRIGAPIMetric(model_name=model, dimension='TA-C')
    metric.task = 't2i'
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(metric.compute(os.path.join(here, "demo.jpg"), "A beautiful sunset with a beach",
                         return_details=True))
