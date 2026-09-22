"""CLIP image and text embeddings, used for the keywords.

Scoring a crop against a list of written prompts gives a label without
training anything, which is what the keywords need. It is not good enough to
tell two supporters apart, so identity comes from `src/faces.py` instead.
`open_clip` and `torch` are imported inside the class so that `--help` stays
instant.
"""

import numpy as np

MODEL = "ViT-B-32"
WEIGHTS = "laion2b_s34b_b79k"


class Encoder:
    """Encodes images and text into one shared, L2-normalised vector space."""

    def __init__(self, device: str, model: str = MODEL, weights: str = WEIGHTS):
        """Load `model` onto `device`, downloading the weights on first use."""
        import open_clip
        import torch

        self._torch = torch
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model, pretrained=weights
        )
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(model)

    def images(self, images) -> np.ndarray:
        """Encode a batch of PIL images into one row each."""
        batch = self._torch.stack([self.preprocess(im) for im in images])
        with self._torch.no_grad():
            vectors = self.model.encode_image(batch.to(self.device))
        return self._normalise(vectors)

    def texts(self, texts: list[str]) -> np.ndarray:
        """Encode a list of prompts into one row each."""
        tokens = self.tokenizer(texts).to(self.device)
        with self._torch.no_grad():
            vectors = self.model.encode_text(tokens)
        return self._normalise(vectors)

    def _normalise(self, vectors) -> np.ndarray:
        vectors = vectors / vectors.norm(dim=-1, keepdim=True)
        return vectors.float().cpu().numpy()
