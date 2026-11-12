import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from knicks_tta import VIEWS, augmix32_probabilities


class RecordingClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(3)
        self.logits = []
        self.batch_sizes = []

    def forward(self, images):
        self.batch_sizes.append(len(images))
        score = 15 * (self.bn(images).mean(dim=(1, 2, 3)) - 0.65)
        logits = torch.stack((score, -score), dim=1)
        self.logits.append(logits.detach().cpu().clone())
        return logits


class InputRecordingClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, images):
        self.inputs.append(images.detach().cpu().clone())
        return torch.zeros((len(images), 2), device=images.device)


class MutatingClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("counter", torch.zeros(()))

    def forward(self, images):
        self.counter.add_(1)
        return torch.zeros((len(images), 2), device=images.device)


class FrozenAugMixTest(unittest.TestCase):
    def setUp(self):
        gradient = torch.arange(3 * 32 * 32, dtype=torch.int64).remainder(256).to(torch.uint8).reshape(3, 32, 32)
        self.images = torch.stack((torch.full_like(gradient, 255), gradient, gradient.flip(-1)))
        self.ids = (101, 303, 505)
        self.mean = torch.zeros(3)
        self.std = torch.ones(3)

    def predict(self, model, images=None, ids=None, batch_size=13):
        return augmix32_probabilities(
            model, self.images if images is None else images,
            self.ids if ids is None else ids, self.mean, self.std, batch_size=batch_size,
        )

    def test_probability_mean_includes_all_views_and_partial_batch(self):
        model = RecordingClassifier()
        original = self.images.clone()
        state = {name: value.clone() for name, value in model.state_dict().items()}
        result = self.predict(model)
        logits = torch.cat(model.logits).reshape(len(self.ids), VIEWS, 2)
        expected = logits.float().softmax(dim=2).mean(dim=1)
        wrong = logits.mean(dim=1).softmax(dim=1)
        self.assertTrue(torch.allclose(result, expected, atol=1e-6, rtol=0))
        self.assertGreater((expected - wrong).abs().max().item(), 1e-4)
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.device.type, "cpu")
        self.assertEqual(sum(model.batch_sizes), len(self.ids) * VIEWS)
        self.assertEqual(model.batch_sizes[-1], (len(self.ids) * VIEWS) % 13)
        self.assertLessEqual(max(model.batch_sizes), 13)
        self.assertTrue(torch.equal(self.images, original))
        self.assertFalse(model.training)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, state[name]), name)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_per_id_predictions_ignore_order_and_chunk_boundaries(self):
        expected = self.predict(RecordingClassifier(), batch_size=13)
        order = (2, 0, 1)
        reordered = self.predict(RecordingClassifier(), self.images[list(order)], tuple(self.ids[i] for i in order), batch_size=17)
        self.assertTrue(torch.allclose(reordered, expected[list(order)], atol=1e-6, rtol=0))
        pieces = [self.predict(RecordingClassifier(), self.images[i:i + 1], self.ids[i:i + 1], batch_size=7) for i in range(3)]
        self.assertTrue(torch.allclose(torch.cat(pieces), expected, atol=1e-6, rtol=0))

    def test_checkpoint_channel_normalization_is_applied_to_the_mixture(self):
        raw_model = InputRecordingClassifier()
        normalized_model = InputRecordingClassifier()
        images, ids = self.images[:1], self.ids[:1]
        mean = torch.tensor([0.2, 0.4, 0.6])
        std = torch.tensor([0.3, 0.5, 0.7])
        self.predict(raw_model, images, ids)
        augmix32_probabilities(normalized_model, images, ids, mean, std, batch_size=11)
        raw = torch.cat(raw_model.inputs)
        normalized = torch.cat(normalized_model.inputs)
        expected = (raw - mean.reshape(1, 3, 1, 1)) / std.reshape(1, 3, 1, 1)
        self.assertTrue(torch.allclose(normalized, expected, atol=1e-6, rtol=0))

    def test_does_not_consume_global_random_streams(self):
        torch_state = torch.random.get_rng_state().clone()
        numpy_state = np.random.get_state()
        self.predict(RecordingClassifier(), self.images[:1], self.ids[:1])
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))
        after = np.random.get_state()
        self.assertEqual(numpy_state[0], after[0])
        np.testing.assert_array_equal(numpy_state[1], after[1])
        self.assertEqual(numpy_state[2:], after[2:])

    def test_detects_buffer_mutation_even_in_eval_mode(self):
        with self.assertRaisesRegex(RuntimeError, "parameters or buffers changed"):
            self.predict(MutatingClassifier(), self.images[:1], self.ids[:1])

    def test_rejects_nonfinite_predictions_instead_of_returning_a_label(self):
        class NonfiniteClassifier(torch.nn.Module):
            def forward(self, images):
                return torch.full((len(images), 2), float("nan"), device=images.device)

        with self.assertRaisesRegex(ValueError, "non-finite probabilities"):
            self.predict(NonfiniteClassifier(), self.images[:1], self.ids[:1])

    def test_rejects_bad_inputs(self):
        model = RecordingClassifier()
        cases = [
            (self.images[:0], (), self.mean, self.std, 2, "nonempty"),
            (self.images.float(), self.ids, self.mean, self.std, 2, "uint8"),
            (self.images[:, :, :31], self.ids, self.mean, self.std, 2, "shape"),
            (self.images, self.ids[:1], self.mean, self.std, 2, "ids"),
            (self.images, (101, None, 505), self.mean, self.std, 2, "ids"),
            (self.images, self.ids, self.mean, self.std, 0, "batch_size"),
            (self.images, self.ids, self.mean, self.std, True, "batch_size"),
            (self.images, self.ids, [0, 0], self.std, 2, "mean and std"),
            (self.images, self.ids, self.mean, [1, 0, 1], 2, "positive std"),
            (self.images, self.ids, self.mean, [1, float("nan"), 1], 2, "finite"),
        ]
        for images, ids, mean, std, batch, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                augmix32_probabilities(model, images, ids, mean, std, batch_size=batch)


if __name__ == "__main__":
    unittest.main()
