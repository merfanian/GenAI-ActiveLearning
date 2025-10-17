from __future__ import annotations

import base64
import io
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import requests
import torch
from PIL import Image
from diffusers import (
    AutoPipelineForInpainting,
    DiffusionPipeline,
    DEISMultistepScheduler,
    FluxFillPipeline,
    StableDiffusionInpaintPipeline,
)
from diffusers.utils import load_image
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(".env")


class AbstractGenerationClient(ABC):

    def __init__(self) -> None:
        self.mask_url = os.getenv("LOCAL_MASK_URL")
        self.label_url = os.getenv("LOCAL_LABEL_URL")
        self.label_mode = os.getenv("LOCAL_LABEL_MODE", "url").lower()
        if self.label_mode == "openai":
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @abstractmethod
    def generate_image(self, prompt: str, guide_image_path: str = None, mask_level: str = "moderate") -> Image.Image:
        pass

    @abstractmethod
    def generate_text(self, prompt: str, guide_text: str = None) -> str:
        pass

    def get_label(
        self, image: Image.Image, prompt: str, label_options: List[str]
    ) -> str:
        if self.label_mode == "human":
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_f:
                try:
                    image.save(temp_f.name)
                    print(f"\n--- Human Labeling Required ---")
                    print(
                        f"An image has been generated. Please review it at: {temp_f.name}"
                    )

                    options_with_discard = label_options + ["discard"]
                    print(f"Labeling prompt: {prompt}")
                    print("Please choose one of the following options:")
                    for i, option in enumerate(options_with_discard):
                        print(f"  {i + 1}: {option}")

                    choice = -1
                    while choice < 1 or choice > len(options_with_discard):
                        try:
                            raw_choice = input(
                                f"Enter your choice (1-{len(options_with_discard)}): "
                            )
                            choice = int(raw_choice)
                        except (ValueError, EOFError):
                            choice = -1
                        if choice < 1 or choice > len(options_with_discard):
                            print("Invalid choice. Please try again.")

                    chosen_option = options_with_discard[choice - 1]
                    logging.info(f"User chose: {chosen_option}")
                    return chosen_option
                finally:
                    os.remove(temp_f.name)

        if self.label_mode == "openai":
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")
            response = self.openai_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{b64_image}",
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()

        with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_image:
            image.save(temp_image, format="JPEG")
            temp_image.seek(0)
            files = {"image": ("image.jpg", temp_image, "image/jpeg")}
            response = requests.post(self.label_url, files=files)
            response.raise_for_status()
            result = response.json()
            try:
                gender = result[0].get("dominant_gender", "").lower()
            except Exception:
                logging.error(f"Unexpected response format: {result}")
                return "discard"

            if gender == "woman":
                return "female"
            return "male"

    def get_text_label(self, text: str, prompt: str, label_options: List[str]) -> str:
        if self.label_mode == "human":
            print(f"\n--- Human Labeling Required ---")
            print(f"Generated text: {text}")
            options_with_discard = label_options + ["discard"]
            print(f"Labeling prompt: {prompt}")
            print("Please choose one of the following options:")
            for i, option in enumerate(options_with_discard):
                print(f"  {i + 1}: {option}")
            choice = -1
            while choice < 1 or choice > len(options_with_discard):
                try:
                    raw_choice = input(
                        f"Enter your choice (1-{len(options_with_discard)}): "
                    )
                    choice = int(raw_choice)
                except (ValueError, EOFError):
                    choice = -1
                if choice < 1 or choice > len(options_with_discard):
                    print("Invalid choice. Please try again.")
            chosen_option = options_with_discard[choice - 1]
            logging.info(f"User chose: {chosen_option}")
            return chosen_option

        if self.label_mode == "openai":
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant for text classification.",
                    },
                    {"role": "user", "content": f'{prompt}\n\nText: "{text}"'},
                ],
            )
            return response.choices[0].message.content.strip()

        # Fallback or other modes can be implemented here
        return "neutral"


class LocalInpaintingClient(AbstractGenerationClient):
    def __init__(self) -> None:
        super().__init__()
        self.inpainting_pipeline = AutoPipelineForInpainting.from_pretrained(
            # "Lykon/dreamshaper-8-inpainting",
            # "OzzyGT/RealVisXL_V4.0_inpainting",
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
            # variant="fp16",
        )
        # self.inpainting_pipeline.scheduler = DEISMultistepScheduler.from_config(self.inpainting_pipeline.scheduler.config)
        self.inpainting_pipeline.to("cuda")

        self.generation_pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        ).to("cuda")

    def generate_text(self, prompt: str, guide_text: str = None) -> str:
        raise NotImplementedError("Local text generation is not supported.")

    def generate_image(self, prompt: str, guide_image_path: str = None, mask_level: str = "moderate") -> Image.Image:
        seed = torch.Generator(device="cuda").seed()
        generator = torch.Generator(device="cuda").manual_seed(seed)
        logging.debug(f"Using seed {seed} for image generation.")

        if guide_image_path:
            img = Image.open(guide_image_path)
            if img.format == "PNG":
                with open(guide_image_path, "rb") as f:
                    img_bytes = f.read()
                filename = os.path.basename(guide_image_path)
            else:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                img_bytes = buf.getvalue()
                filename = Path(guide_image_path).with_suffix(".png").name

            mask_resp = requests.post(
                self.mask_url + f"?accuracy_level={mask_level}&mask_type=black_and_white",
                files={"image": (filename, img_bytes, "image/png")},
            )
            mask_resp.raise_for_status()
            mask_bytes = mask_resp.content
            mask_image = Image.open(io.BytesIO(mask_bytes)).convert("RGB")
            mask_image.save(f"masks/{filename}", format="PNG")

            init_image = load_image(guide_image_path)
            generated_image = self.inpainting_pipeline(
                prompt=prompt,
                image=init_image.resize((1024, 1024)),
                mask_image=mask_image.resize((1024, 1024)),
                generator=generator,
                guidance_scale=8.0,
                num_inference_steps=30,  # steps between 15 and 30 work well for us
                strength=0.99,
            ).images[0]
            return generated_image
        else:
            # From-scratch generation using a blank image and a white mask
            init_image = Image.new("RGB", (1024, 1024), "black")
            mask_image = Image.new("RGB", (1024, 1024), "white")

            generated_image = self.generation_pipeline(
                prompt=prompt,
                generator=generator,
            ).images[0]
            return generated_image


class OpenAIGenerationClient(AbstractGenerationClient):
    def __init__(self) -> None:
        super().__init__()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_text(self, prompt: str, guide_text: str = None) -> str:
        raise NotImplementedError(
            "OpenAI text generation is not implemented in this client."
        )

    def generate_image(self, prompt: str, guide_image_path: str = None, mask_level: str = "moderate") -> Image.Image:
        if guide_image_path:
            img = Image.open(guide_image_path)
            if img.format == "PNG":
                with open(guide_image_path, "rb") as f:
                    img_bytes = f.read()
                filename = os.path.basename(guide_image_path)
            else:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                img_bytes = buf.getvalue()
                filename = Path(guide_image_path).with_suffix(".png").name

            mask_resp = requests.post(
                self.mask_url + f"?accuracy_level={mask_level}&mask_type=transparent",
                files={"image": (filename, img_bytes, "image/png")},
            )
            mask_resp.raise_for_status()
            mask_bytes = mask_resp.content
            mask_image = Image.open(io.BytesIO(mask_bytes)).convert("RGB")
            mask_image.save(f"masks/{filename}", format="PNG")

            for attempt in range(1, 4):
                try:
                    response = self.client.images.edit(
                        image=(filename, img_bytes, "image/png"),
                        prompt=prompt,
                        mask=(filename, mask_bytes, "image/png"),
                        n=1,
                        model="dall-e-2",
                        size="1024x1024",
                        response_format="b64_json",
                    )
                    logging.info(
                        f"Used images.edit with guide image on attempt {attempt}"
                    )
                    image_base64 = response.data[0].b64_json
                    return Image.open(io.BytesIO(base64.b64decode(image_base64)))
                except Exception as e:
                    logging.error(f"Attempt {attempt} failed: {e}")
                    if attempt == 3:
                        logging.error(
                            f"Failed to generate image with guide {guide_image_path} after 3 attempts."
                        )
                        return None
        else:
            for attempt in range(1, 4):
                try:
                    response = self.client.images.generate(
                        prompt=prompt,
                        n=1,
                        model="dall-e-2",
                        size="1024x1024",
                        response_format="b64_json",
                    )
                    logging.info(
                        f"Used images.edit without guide image on attempt {attempt}"
                    )
                    image_base64 = response.data[0].b64_json
                    return Image.open(io.BytesIO(base64.b64decode(image_base64)))
                except Exception as e:
                    logging.error(f"Attempt {attempt} failed: {e}")
                    if attempt == 3:
                        logging.error(
                            "Failed to generate image without guide image after 3 attempts."
                        )
                        return None


class OpenAITextGenerationClient(AbstractGenerationClient):
    def __init__(self) -> None:
        super().__init__()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_image(self, prompt: str, guide_image_path: str = None) -> Image.Image:
        raise NotImplementedError(
            "Image generation is not supported in the text client."
        )

    def generate_text(self, prompt: str, guide_text: str = None) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that generates text samples for sentiment analysis.",
            },
            {"role": "user", "content": prompt},
        ]
        if guide_text:
            messages.append(
                {"role": "assistant", "content": f"Here is an example: {guide_text}"}
            )

        for attempt in range(1, 4):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo", messages=messages, n=1, temperature=0.7
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logging.error(f"Attempt {attempt} failed: {e}")
                if attempt == 3:
                    logging.error("Failed to generate text after 3 attempts.")
                    return None


def get_generation_client(modality: str = "image") -> AbstractGenerationClient:
    provider = os.getenv("GENERATION_PROVIDER", "openai").lower()

    if modality == "image":
        if provider == "local_inpainting":
            logging.debug("Using LocalInpaintingClient for image generation")
            return LocalInpaintingClient()
        logging.debug("Using OpenAIGenerationClient for image generation")
        return OpenAIGenerationClient()
    elif modality == "text":
        logging.debug("Using OpenAITextGenerationClient for text generation")
        return OpenAITextGenerationClient()
    else:
        raise ValueError(f"Unsupported modality: {modality}")
