"""Tests for `src.faces`, without loading the model."""

import numpy as np

from src import faces


class _Detection:
    """Stands in for one insightface detection."""

    def __init__(self, bbox, det_score: float):
        self.bbox = bbox
        self.det_score = det_score


class TestProviders:
    """Choosing which onnxruntime backends to offer."""

    def test_cpu_is_always_offered(self):
        """Every machine can run the model, even with no accelerator."""
        available = ["CPUExecutionProvider"]
        assert faces.providers(available, "mps") == ["CPUExecutionProvider"]

    def test_prefers_an_accelerator(self):
        """An available accelerator is tried before the CPU."""
        available = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        assert faces.providers(available, "mps")[0] == "CoreMLExecutionProvider"

    def test_ignores_providers_that_are_not_available(self):
        """Naming a missing provider makes onnxruntime complain, so do not."""
        chosen = faces.providers(["CPUExecutionProvider"], "0")
        assert "CUDAExecutionProvider" not in chosen

    def test_cuda_outranks_coreml(self):
        """A machine with both is a desktop GPU, which is the faster path."""
        available = ["CoreMLExecutionProvider", "CUDAExecutionProvider"]
        assert faces.providers(available, "0")[0] == "CUDAExecutionProvider"

    def test_device_cpu_turns_the_accelerator_off(self):
        """`--device cpu` has to mean the same thing for both models."""
        available = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert faces.providers(available, "cpu") == ["CPUExecutionProvider"]


class TestNearest:
    """Deciding which detected face the crop is actually of."""

    def test_no_detections(self):
        """A crop of a hand or a banner holds no face at all."""
        assert faces.nearest([], (0, 0, 10, 10)) is None

    def test_picks_the_face_at_the_recorded_head(self):
        """A neighbour caught at the edge is not the person we cropped for."""
        wanted = _Detection(np.array([10, 10, 50, 50]), 0.6)
        neighbour = _Detection(np.array([200, 10, 240, 50]), 0.9)
        assert faces.nearest([neighbour, wanted], (12, 12, 48, 48)) is wanted

    def test_falls_back_to_the_surest_detection(self):
        """Without a head box there is nothing to measure against."""
        weak = _Detection(np.array([10, 10, 50, 50]), 0.4)
        strong = _Detection(np.array([200, 10, 240, 50]), 0.9)
        assert faces.nearest([weak, strong], None) is strong


class TestGender:
    """The mapping from the model's own labels."""

    def test_names_both_sexes(self):
        """The keyword is a word, not the single letter the model returns."""
        assert faces.GENDER["M"] == "man"
        assert faces.GENDER["F"] == "woman"
