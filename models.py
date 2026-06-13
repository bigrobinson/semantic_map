import os
import json
import hashlib
import base64
import io
import requests
import numpy as np
from PIL import Image
from typing import List, Union

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".embedding_cache.json")

class GeminiEmbeddingModel:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.cache = {}
        self._load_cache()
        if not self.api_key:
            print("Warning: GEMINI_API_KEY not found in environment. Using deterministic offline mock embeddings.")

    def _load_cache(self):
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r") as f:
                    self.cache = json.load(f)
            except Exception as e:
                print(f"Error loading embedding cache: {e}")
                self.cache = {}

    def _save_cache(self):
        try:
            with open(CACHE_FILE, "w") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Error saving embedding cache: {e}")

    def _get_mock_embedding(self, text: str, dim: int = 768) -> List[float]:
        """Generates a deterministic pseudo-random unit vector for a given text."""
        hash_obj = hashlib.sha256(text.encode("utf-8"))
        seed = int(hash_obj.hexdigest(), 16) % (2**32 - 1)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(dim)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()

    def embed_text(self, text: str) -> List[float]:
        """Retrieves the 768-dimensional embedding for a string, using cache/API/mock."""
        if not text:
            return [0.0] * 768
            
        cache_key = f"text:{text}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        if not self.api_key:
            embedding = self._get_mock_embedding(text)
        else:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent?key={self.api_key}"
                headers = {"Content-Type": "application/json"}
                payload = {
                    "model": "models/gemini-embedding-2",
                    "content": {
                        "parts": [{"text": text}]
                    },
                    "outputDimensionality": 768
                }
                response = requests.post(url, headers=headers, json=payload, timeout=10)
                if response.status_code == 200:
                    embedding = response.json()["embedding"]["values"]
                else:
                    print(f"API Error ({response.status_code}): {response.text}. Falling back to mock embedding.")
                    embedding = self._get_mock_embedding(text)
            except Exception as e:
                print(f"Error calling Gemini API for text embedding: {e}. Falling back to mock embedding.")
                embedding = self._get_mock_embedding(text)

        self.cache[cache_key] = embedding
        self._save_cache()
        return embedding

    def generate_image_description(self, img_array: np.ndarray) -> str:
        """Generates a text description of a numpy image using gemini-2.5-flash."""
        if not self.api_key:
            # Deterministic mock description based on image content hash
            img_hash = hashlib.sha256(img_array.tobytes()).hexdigest()[:8]
            return f"Mock description of image {img_hash} (offline mode)"

        # Compute hash to check cache first
        img_hash = hashlib.sha256(img_array.tobytes()).hexdigest()
        cache_key = f"desc_image:{img_hash}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            # Convert numpy array to PNG bytes
            img = Image.fromarray(img_array.astype('uint8'))
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')

            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{
                    "parts": [
                        {"text": "Describe this image in detail for a robot semantic map. Focus on objects, layout, and labels."},
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": img_b64
                            }
                        }
                    ]
                }]
            }
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            if response.status_code == 200:
                description = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            else:
                print(f"API Error ({response.status_code}): {response.text}")
                description = f"Image description failed (Error {response.status_code})"
        except Exception as e:
            print(f"Error calling Gemini API for image description: {e}")
            description = "Image description failed due to exception"

        self.cache[cache_key] = description
        self._save_cache()
        return description

    def embed_image(self, img_array: np.ndarray) -> List[float]:
        """Retrieves a 768-dimensional embedding for an image."""
        description = self.generate_image_description(img_array)
        return self.embed_text(description)

# Singleton function matching requirements
_model_instance = None

def embedding_model() -> GeminiEmbeddingModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = GeminiEmbeddingModel()
    return _model_instance
