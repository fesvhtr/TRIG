"""The TRIGScore judge: one place for the rating scale, the judge prompts, endpoint resolution
and the client. Everything that talks to the judge (trig.metrics.trig_api, scripts/diag,
trig_rl.rewards) imports from here.

Protocols (text-to-image):
  baseline  the paper's prompts (trig/config.py), prompt text + image for every dimension
  v1        same text, but image-quality dimensions (IQ-*) are judged from the image alone
  v2        rewritten prompts (trig/config_judge_v2.py): anchored scale, "what does NOT count"
"""
from __future__ import annotations

import hashlib
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

from PIL import Image

from trig import config_judge_v2
from trig.config import gpt_logit_dimension_msg, gpt_logit_system_msg
from trig.utils.utils import image_data_url

# ----------------------------------------------------------------------------- rating scale
RATING_WORDS = ["terrible", "bad", "medium", "good", "excellent"]   # worst -> best
RATING_WEIGHTS = {w: i / (len(RATING_WORDS) - 1) for i, w in enumerate(RATING_WORDS)}
IMAGE_ONLY_DIMS = config_judge_v2.IMAGE_ONLY                          # {"IQ-R", "IQ-O", "IQ-A"}
DIMS, EXCLUDED_DIMS, pair_allowed = config_judge_v2.DIMS, config_judge_v2.EXCLUDED_DIMS, config_judge_v2.pair_allowed
PROTOCOLS = ("baseline", "v1", "v2")
_LEADING_SPACE_MARKS = ("Ġ", "▁", " ", "_")


def normalize_token(token: str) -> str:
    t = token
    while t and t[0] in _LEADING_SPACE_MARKS:
        t = t[1:]
    return "".join(ch for ch in t.lower() if ch.isalpha())


def token_to_rating_word(token: str) -> Optional[str]:
    """Prefix-match a (possibly partial) token to a rating word: ``ex`` -> excellent,
    ``ter`` -> terrible, ``good.`` -> good. Single letters are ambiguous and ignored."""
    n = normalize_token(token)
    if len(n) < 2:
        return None
    m = [w for w in RATING_WORDS if w.startswith(n) or n.startswith(w)]
    return m[0] if len(m) == 1 else None


def logprobs_to_score(top_logprobs) -> Tuple[float, Dict[str, float]]:
    """Expected rating in [0, 1] and the normalised word probabilities, from objects with
    ``.token`` / ``.logprob`` (the first generated token's top-logprobs)."""
    probs: Dict[str, float] = {}
    for item in top_logprobs:
        w = token_to_rating_word(item.token)
        if w is None:
            continue
        try:
            p = math.exp(float(item.logprob))
        except OverflowError:
            p = 0.0
        probs[w] = probs.get(w, 0.0) + p
    if not probs:
        return 0.0, {}
    z = sum(probs.values()) + 1e-10
    probs = {w: p / z for w, p in probs.items()}
    return sum(RATING_WEIGHTS[w] * p for w, p in probs.items()), probs


# ----------------------------------------------------------------------------- prompts
_V1_IMAGE_ONLY_SYSTEM = '''
You are an evaluation assistant. I will give you an AI generated image. Evaluate ONLY the following dimension of the image itself; you are deliberately not shown the prompt, so judge purely from what is visible.
The information and evaluation criteria about the dimension is as follows:
{}
1. Judge the image on this dimension only; do not guess what was requested.
2. You should evaluate the image in this dimension by a scale from: excellent, good, medium, bad, terrible.
Your grading scale should be uniform; Excellent for accuracy, Good for very good performance, Medium for acceptable, Bad for some errors, and Terrible for more errors.
3. You must give me one of these words as your evaluation, your answer should only be one word.
'''
_PROMPT_LABEL = {"t2i": "Prompt", "p2p": "Editing prompt", "s2p": "Subject-driven Prompt"}


def build_messages(protocol: str, dim: str, prompt: str, image_url: str, task: str = "t2i",
                   item: Optional[str] = None) -> list:
    """OpenAI chat messages for one (image, dimension) judgement."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown judge protocol {protocol!r}; expected one of {PROTOCOLS}")
    image_part = {"type": "image_url", "image_url": {"url": image_url}}
    criteria = gpt_logit_dimension_msg[task][dim]
    if protocol == "v2":
        if task != "t2i":
            raise ValueError("protocol v2 is defined for t2i only")
        return [{"role": "system", "content": config_judge_v2.system_message(dim)},
                {"role": "user", "content": [image_part, {"type": "text", "text": config_judge_v2.user_text(dim, prompt)}]}]
    if protocol == "v1" and dim in IMAGE_ONLY_DIMS:
        return [{"role": "system", "content": _V1_IMAGE_ONLY_SYSTEM.format(criteria)},
                {"role": "user", "content": [{"type": "text", "text": "Evaluate this image."}, image_part]}]
    system = gpt_logit_system_msg[task].format(*([item, criteria] if task == "s2p" else [criteria]))
    text = f"{_PROMPT_LABEL[task]} for generating this image: {prompt}"
    return [{"role": "system", "content": system},
            {"role": "user", "content": [{"type": "text", "text": text}, image_part]}]


# ----------------------------------------------------------------------------- endpoint
@dataclass(frozen=True)
class JudgeConfig:
    endpoint: str
    model: str
    api_key: str = "EMPTY"


def resolve_judge(endpoint: Optional[str] = None, model: Optional[str] = None,
                  api_key: Optional[str] = None) -> JudgeConfig:
    """Precedence: explicit argument > TRIG_JUDGE_* environment variable > default."""
    return JudgeConfig(
        endpoint=endpoint or os.environ.get("TRIG_JUDGE_ENDPOINT", "http://127.0.0.1:8000/v1/"),
        model=model or os.environ.get("TRIG_JUDGE_MODEL", "Qwen/Qwen3.5-9B"),
        api_key=api_key or os.environ.get("TRIG_JUDGE_API_KEY", "EMPTY"),
    )


# ----------------------------------------------------------------------------- client
ImageLike = Union[str, Image.Image]


class JudgeClient:
    """Scores (image, prompt, dim) items in parallel through an OpenAI-compatible server.

    One forward pass per item (first-token logprobs), thinking disabled for Qwen3 chat
    templates, a score cache keyed by image content + dim + protocol, and a hard failure when
    any item could not be scored: a dead judge must stop a run, not reward everything 0.
    """

    def __init__(self, endpoint: Optional[str] = None, model: Optional[str] = None, api_key: Optional[str] = None,
                 protocol: str = "v2", task: str = "t2i", top_logprobs: int = 10, max_concurrent: int = 32,
                 max_retries: int = 3, timeout: float = 120.0, max_side: Optional[int] = None,
                 max_cache_size: int = 50000):
        import openai

        self.cfg = resolve_judge(endpoint, model, api_key)
        self.client = openai.Client(api_key=self.cfg.api_key, base_url=self.cfg.endpoint, timeout=timeout, max_retries=0)
        self.protocol, self.task, self.top_logprobs = protocol, task, top_logprobs
        self.max_concurrent, self.max_retries, self.max_side = max_concurrent, max_retries, max_side
        self._cache: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._max_cache = max_cache_size

    @property
    def model(self) -> str:
        return self.cfg.model

    def _extra_body(self):
        # Qwen3/Qwen3.5 templates honour enable_thinking; OpenAI rejects unknown fields.
        return None if "gpt" in self.cfg.model.lower() else {"chat_template_kwargs": {"enable_thinking": False}}

    def score_one(self, image_url: str, prompt: str, dim: str, item: Optional[str] = None,
                  return_details: bool = False):
        """Score one already-encoded image. Raises after ``max_retries`` failures."""
        messages = build_messages(self.protocol, dim, prompt, image_url, task=self.task, item=item)
        err = None
        for attempt in range(self.max_retries):
            try:
                c = self.client.chat.completions.create(
                    model=self.cfg.model, messages=messages, logprobs=True, top_logprobs=self.top_logprobs,
                    temperature=1.0, max_tokens=8, extra_body=self._extra_body())
                first = c.choices[0].logprobs.content[0]
                score, probs = logprobs_to_score(first.top_logprobs)
                return (score, {"first_token": first.token, "word_probs": probs}) if return_details else score
            except Exception as e:  # transient server errors
                err = e
                if attempt + 1 < self.max_retries:
                    import time
                    time.sleep(0.5 * 2 ** attempt)
        raise RuntimeError(f"judge failed after {self.max_retries} attempts ({self.cfg.model} @ {self.cfg.endpoint}): {err}")

    @staticmethod
    def _image_key(img: ImageLike) -> str:
        if isinstance(img, str):
            return os.path.abspath(img)
        return hashlib.md5(img.tobytes()).hexdigest() + f"_{img.size[0]}x{img.size[1]}"

    def score(self, items: Iterable[Tuple[ImageLike, str, str]]) -> List[float]:
        """items: (image path or PIL image, prompt, dim). One encode per distinct image."""
        items = list(items)
        urls: Dict[str, str] = {}
        keys = []
        for img, _, _ in items:
            k = self._image_key(img)
            keys.append(k)
            if k not in urls:
                urls[k] = image_data_url(img, max_side=self.max_side)

        def run(i):
            img, prompt, dim = items[i]
            ck = f"{keys[i]}|{dim}|{self.protocol}"
            with self._lock:
                if ck in self._cache:
                    return self._cache[ck]
            s = self.score_one(urls[keys[i]], prompt, dim)
            with self._lock:
                if len(self._cache) >= self._max_cache:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[ck] = s
            return s

        with ThreadPoolExecutor(max_workers=max(1, self.max_concurrent)) as pool:
            return list(pool.map(run, range(len(items))))
