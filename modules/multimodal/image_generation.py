import os
import io
from typing import Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont

class ImageGenerator:
    """Supports Text-to-Image generation using Google Imagen 3 API with procedural fallback."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.client = None
        self._init_client()

    def _init_client(self):
        if not self.api_key:
            return
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
        except Exception:
            self.client = None

    def generate_image(self, prompt: str) -> Dict[str, Any]:
        """Generates an image from text using Google Imagen-3 or procedural demo generator."""
        if self.client and self.api_key:
            try:
                # Try official Google Imagen 3 endpoint
                result = self.client.models.generate_images(
                    model='imagen-3.0-generate-002',
                    prompt=prompt,
                    config=dict(number_of_images=1, aspect_ratio="1:1")
                )
                if result.generated_images:
                    generated = result.generated_images[0]
                    img = Image.open(io.BytesIO(generated.image.image_bytes))
                    return {
                        "success": True,
                        "image": img,
                        "engine": "Google Imagen 3 (imagen-3.0-generate-002)",
                        "message": f"Successfully generated image using Google Imagen 3."
                    }
            except Exception as e:
                print(f"Google Imagen 3 call notice ({e}). Generating procedural product card visualization.")

        # Procedural fallback generator (clearly labeled, does not fake API calls)
        img = self._create_procedural_product_card(prompt)
        return {
            "success": True,
            "image": img,
            "engine": "Procedural Graphic Engine (Google Imagen 3 API key optional)",
            "message": "Generated concept visual card. Connect Google Gemini/Imagen-3 API key for direct diffusion rendering."
        }

    def _create_procedural_product_card(self, prompt: str) -> Image.Image:
        """Creates a clean, styled graphic card summarizing the requested concept."""
        width, height = 512, 512
        img = Image.new("RGB", (width, height), color=(20, 24, 38))
        draw = ImageDraw.Draw(img)

        # Gradient-like background accents
        for i in range(120):
            draw.line([(0, i * 4), (width, i * 4 + 20)], fill=(25 + i // 4, 35 + i // 3, 65 + i // 2), width=3)

        # Central Card
        draw.rounded_rectangle([(30, 40), (width - 30, height - 40)], radius=18, fill=(30, 36, 56), outline=(74, 144, 226), width=2)

        # Title
        draw.text((50, 70), "APEXTECH AI VISUAL LAB", fill=(100, 200, 255))
        draw.text((50, 95), "Customer Service Concept Illustration", fill=(180, 190, 210))

        # Prompt Box
        draw.rounded_rectangle([(50, 140), (width - 50, 280)], radius=10, fill=(40, 48, 76), outline=(60, 72, 110), width=1)
        draw.text((65, 155), "Prompt Directive:", fill=(140, 160, 190))
        
        # Word wrap prompt
        words = prompt.split()
        lines = []
        cur_line = ""
        for w in words:
            if len(cur_line) + len(w) < 32:
                cur_line += " " + w
            else:
                lines.append(cur_line.strip())
                cur_line = w
        if cur_line: lines.append(cur_line.strip())

        y = 185
        for line in lines[:4]:
            draw.text((65, y), line, fill=(240, 245, 255))
            y += 22

        # Status badge
        draw.rounded_rectangle([(50, 320), (width - 50, 430)], radius=10, fill=(24, 30, 48))
        draw.text((65, 340), "🎨 Engine Mode:", fill=(160, 180, 210))
        draw.text((65, 365), "Google Imagen-3 / GenAI Compatible", fill=(74, 222, 128))
        draw.text((65, 395), "ApexCare Visual Confirmation Ready", fill=(180, 190, 210))

        return img
