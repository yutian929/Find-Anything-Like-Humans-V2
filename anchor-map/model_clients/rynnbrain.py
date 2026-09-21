"""RynnBrain 客户端：本地(torch) + 远程(HTTP)。"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

from PIL import Image

from .base import RynnBrainBase, RemoteMixin, image_to_b64

logger = logging.getLogger("model_clients.rynnbrain")

_BBOX_PATTERN = re.compile(
    r"<object>.*?\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\).*?</object>", re.DOTALL,
)


def _existence_prompt(obj: str) -> str:
    return (
        f"Verify if the EXACT '{obj}' is present in this image.\n"
        "Use this strict checklist:\n"
        "- Color matches exactly? (If no -> 'No')\n"
        "- 3D Shape matches exactly? (If flat instead of 3D -> 'No')\n"
        "- Category matches exactly? (If look-alike -> 'No')\n"
        "Answer 'No' if ANY check fails. Output exactly one word: 'Yes' or 'No'."
        "Be careful not to mistake the '药盒' for '瓶装矿泉水'."
    )


def _bbox_prompt(obj: str) -> str:
    return (
        f"Localize the EXACT '{obj}' in this image.\n"
        "RULES:\n"
        "1. MUST match exact color, 3D shape, and category.\n"
        "2. Tightly enclose ONLY the true target.\n"
        "Generate coordinates for exactly one object. x1,y1,x2,y2 in [0,1000].\n"
        "Output strictly: <object> (x1, y1), (x2, y2) </object>"
    )


def _parse_bbox(answer: str) -> Optional[List[int]]:
    m = _BBOX_PATTERN.search(answer or "")
    return [int(m.group(i)) for i in range(1, 5)] if m else None


class RynnBrainLocal(RynnBrainBase):
    def __init__(self, path: str | Path, device: str = "auto", batch_size: int = 4):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        import torch

        self.torch = torch
        self.batch_size = int(batch_size)
        self.model_path = Path(path)
        logger.info("Loading RynnBrain (local): %s", self.model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(self.model_path), dtype="auto", device_map=device,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(str(self.model_path))
        try:
            self.processor.tokenizer.padding_side = "left"  # 批量 generate 必须左侧 padding
        except AttributeError:
            pass

    def _generate_batch(self, images: List[Image.Image], prompt: str, max_new_tokens: int = 128) -> List[str]:
        if not images:
            return []
        messages_list = [
            [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
            for img in images
        ]
        inputs = self.processor.apply_chat_template(
            messages_list, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", padding=True,
        ).to(self.model.device)
        with self.torch.no_grad():
            resp = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, resp)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def _generate_text_only(self, prompt: str, max_new_tokens: int = 64) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        with self.torch.no_grad():
            resp = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, resp)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def _run_chunked(self, images: List[Image.Image], prompt: str) -> List[str]:
        chunk = len(images) if self.batch_size <= 0 else self.batch_size
        out: List[str] = []
        for i in range(0, len(images), chunk):
            out.extend(self._generate_batch(images[i:i + chunk], prompt))
        return out

    def exists_batch(self, images: List[Image.Image], obj: str) -> List[bool]:
        answers = self._run_chunked(images, _existence_prompt(obj))
        for i, ans in enumerate(answers):
            logger.info("Existence[%s] #%d: %s", obj, i, ans)
        return ["yes" in ans.lower() for ans in answers]

    def bbox_batch(self, images: List[Image.Image], obj: str) -> List[Optional[List[int]]]:
        answers = self._run_chunked(images, _bbox_prompt(obj))
        return [_parse_bbox(ans) for ans in answers]

    def translate(self, zh: str) -> str:
        source = str(zh).strip()
        prompt = (
            "Translate the following Chinese object/category name into English. "
            "Output only the concise English translation, without explanations, "
            f"quotation marks, punctuation, or extra words: {source}"
        )
        answer = self._generate_text_only(prompt, max_new_tokens=64).strip().splitlines()[0].strip()
        answer = re.sub(r"^(translation|english)\s*:\s*", "", answer, flags=re.IGNORECASE)
        answer = answer.strip(" \t\r\n\"'`.,;:!?")
        if not answer or len(answer) > 160 or re.search(r"[一-鿿]", answer):
            raise ValueError(f"RynnBrain returned invalid English translation: {answer!r}")
        logger.info("Translate[%s] -> %s", source, answer)
        return answer


class RynnBrainRemote(RemoteMixin, RynnBrainBase):
    def __init__(self, url: str, timeout: float = 30.0, batch_size: int = 4):
        RemoteMixin.__init__(self, url, timeout)
        self.batch_size = int(batch_size)

    def exists_batch(self, images: List[Image.Image], obj: str) -> List[bool]:
        if not images:
            return []
        data = self._post("/exists_batch", {"images_b64": [image_to_b64(i) for i in images], "object": obj})
        return [bool(v) for v in data.get("exists", [])]

    def bbox_batch(self, images: List[Image.Image], obj: str) -> List[Optional[List[int]]]:
        if not images:
            return []
        data = self._post("/bbox_batch", {"images_b64": [image_to_b64(i) for i in images], "object": obj})
        out: List[Optional[List[int]]] = []
        for b in data.get("bboxes", []):
            out.append([int(v) for v in b] if b else None)
        return out

    def translate(self, zh: str) -> str:
        return str(self._post("/translate", {"text": str(zh)}).get("english", "")).strip()
