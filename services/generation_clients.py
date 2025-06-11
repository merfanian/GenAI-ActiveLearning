from __future__ import annotations

import os
import base64
import io
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import requests
from dotenv import load_dotenv
from PIL import Image
from openai import OpenAI
import torch
import torchvision.transforms as transforms

from services.model_service import SimpleCNN
from utils.config import TRAINED_MODELS_DIR

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
        self.mask_url = os.getenv("LOCAL_MASK_URL")
        self.generate_url = os.getenv("LOCAL_GENERATE_URL")
        self.label_url = os.getenv("LOCAL_LABEL_URL")
        self.label_mode = os.getenv("LOCAL_LABEL_MODE", "url").lower()
        if self.label_mode == "model":
            model_path = TRAINED_MODELS_DIR / "perfect.pth"
            checkpoint = torch.load(model_path, map_location="cpu")
            self.idx_to_label = checkpoint["idx_to_label"]
            self.model = SimpleCNN(num_classes=len(self.idx_to_label))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
        elif self.label_mode == "openai":
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_image(self, guide_image_path: str, prompt: str) -> Image.Image:
        # --- 1. Load & (conditionally) convert to PNG ---
        img = Image.open(guide_image_path)
        if img.format == "PNG":
            # already PNG → just read bytes
            with open(guide_image_path, "rb") as f:
                img_bytes = f.read()
            filename = os.path.basename(guide_image_path)
        else:
            # convert to PNG in-memory
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            filename = Path(guide_image_path).with_suffix(".png").name

        # --- 2. Get mask from your mask-service endpoint ---
        mask_resp = requests.post(
            self.mask_url,
            files={
                # tell FastAPI/OpenAI this is a PNG
                "image": (filename, img_bytes, "image/png")
            },
        )
        mask_resp.raise_for_status()
        mask_bytes = mask_resp.content

        # --- 3. Call the image-edit endpoint (always PNG) ---
        gen_resp = requests.post(
            self.generate_url,
            files={
                "image": (filename, img_bytes, "image/png"),
                "mask": ("mask.png", mask_bytes, "image/png"),
            },
            data={
                "prompt": prompt,
                "n": "1",
                "size": "512x512",
            },
        )
        gen_resp.raise_for_status()

        # --- 4. Decode & return PIL image ---
        resp_json = gen_resp.json()
        img_b64 = resp_json["data"][0]["b64_json"]
        out_bytes = base64.b64decode(img_b64)
        return Image.open(io.BytesIO(out_bytes))

    def get_label(self, image: Image.Image, prompt: str, label_options: List[str]) -> str:
        if self.label_mode == "model":
            with torch.no_grad():
                inp = self.transform(image).unsqueeze(0)
                logits = self.model(inp)
                pred_idx = int(torch.argmax(logits, dim=1).item())
                return self.idx_to_label[pred_idx]
        elif self.label_mode == "openai":
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")
            response = self.openai_client.responses.create(
                model="gpt-4.1",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{b64_image}",
                            },
                        ],
                    }
                ],
            )
            return response.output_text.strip()
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
    provider = os.getenv("GENERATION_PROVIDER", "local").lower()
    if provider == "local":
        logging.debug("Using LocalGenerationClient for image generation")
        return LocalGenerationClient()
    logging.debug("Using OpenAIGenerationClient for image generation")
    return OpenAIGenerationClient()
