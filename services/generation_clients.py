from __future__ import annotations

import os
import base64
import io
import logging
from abc import ABC, abstractmethod
from typing import List

import requests
from dotenv import load_dotenv
from PIL import Image
from openai import OpenAI

load_dotenv(".env")


class AbstractGenerationClient(ABC):
    @abstractmethod
    def generate_image(self, guide_image_path: str, prompt: str) -> Image.Image:
        pass

    @abstractmethod
    def get_label(self, image: Image.Image, prompt: str, label_options: List[str]) -> str:
        pass


class OpenAIGenerationClient(AbstractGenerationClient):
    def __init__(self) -> None:
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_image(self, guide_image_path: str, prompt: str) -> Image.Image:
        with open(guide_image_path, "rb") as guide_file:
            try:
                response = self.client.images.edit(
                    image=[guide_file], prompt=prompt, n=1, model="gpt-image-1"
                )
                logging.debug("Used generate_edit for image generation")
            except Exception as e:
                logging.error(f"Failed to generate edit for image generation: {e}")
                raise
        image_base64 = response.data[0].b64_json
        return Image.open(io.BytesIO(base64.b64decode(image_base64)))

    def get_label(self, image: Image.Image, prompt: str, label_options: List[str]) -> str:
        chat_resp = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return chat_resp.choices[0].message.content.strip()


class LocalGenerationClient(AbstractGenerationClient):
    def __init__(self) -> None:
        mask_url = os.getenv("LOCAL_MASK_URL")
        generate_url = os.getenv("LOCAL_GENERATE_URL")
        label_url = os.getenv("LOCAL_LABEL_URL")

        if not (mask_url and generate_url and label_url):
            base_url = os.getenv("LOCAL_GEN_API_URL", "http://localhost")
            port = os.getenv("LOCAL_GEN_PORT")
            if port:
                base_url = f"{base_url}:{port}"
            self.mask_url = os.getenv(
                "LOCAL_MASK_URL",
                f"{base_url}{os.getenv('LOCAL_MASK_ENDPOINT', '/v1/images/masks')}"
            )
            self.gen_url = os.getenv(
                "LOCAL_GENERATE_URL",
                f"{base_url}{os.getenv('LOCAL_GENERATE_ENDPOINT', '/v1/images/edits')}"
            )
            self.label_url = os.getenv(
                "LOCAL_LABEL_URL",
                f"{base_url}{os.getenv('LOCAL_LABEL_ENDPOINT', '/v1/images/labels')}"
            )
        else:
            self.mask_url = mask_url
            self.gen_url = generate_url
            self.label_url = label_url

    def generate_image(self, guide_image_path: str, prompt: str) -> Image.Image:
        with open(guide_image_path, "rb") as f:
            img_bytes = f.read()

        mask_resp = requests.post(
            self.mask_url,
            files={"image": (os.path.basename(guide_image_path), img_bytes, "image/png")},
        )
        mask_resp.raise_for_status()
        mask_bytes = mask_resp.content

        gen_resp = requests.post(
            self.gen_url,
            files={
                "image": ("image.png", img_bytes, "image/png"),
                "mask": ("mask.png", mask_bytes, "image/png"),
            },
            data={"prompt": prompt},
        )
        gen_resp.raise_for_status()
        resp_json = gen_resp.json()
        img_b64 = resp_json["data"][0]["b64_json"]
        return Image.open(io.BytesIO(base64.b64decode(img_b64)))

    def get_label(self, image: Image.Image, prompt: str, label_options: List[str]) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        resp = requests.post(
            self.label_url,
            files={"image": ("image.png", buf.getvalue(), "image/png")},
            data={"prompt": prompt},
        )
        resp.raise_for_status()
        return resp.json().get("label", "").strip()


def get_generation_client() -> AbstractGenerationClient:
    provider = os.getenv("GENERATION_PROVIDER", "openai").lower()
    if provider == "local":
        logging.debug("Using LocalGenerationClient for image generation")
        return LocalGenerationClient()
    logging.debug("Using OpenAIGenerationClient for image generation")
    return OpenAIGenerationClient()
