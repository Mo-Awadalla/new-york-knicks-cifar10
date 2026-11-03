import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from knicks_cifar import augment


class CutoutBoundaryTest(unittest.TestCase):
    def test_masks_are_centered_anywhere_and_clipped_at_image_edges(self):
        images = torch.full((128, 3, 32, 32), 255, dtype=torch.uint8)
        mean = torch.zeros(3)
        std = torch.ones(3)
        centers = torch.randint(0, 32, (128, 2), generator=torch.Generator().manual_seed(3))
        expected = torch.ones((128, 3, 32, 32))
        for image, (y, x) in zip(expected, centers.tolist()):
            image[:, max(0, y - 8):min(32, y + 8), max(0, x - 8):min(32, x + 8)] = 0
        actual = augment(images, "cutout", mean, std, torch.Generator().manual_seed(2), torch.Generator().manual_seed(3))
        self.assertTrue(torch.equal(actual, expected))


class RandAugmentIndependenceTest(unittest.TestCase):
    def test_identical_images_can_receive_different_unmasked_transformations(self):
        images = torch.full((32, 3, 32, 32), 127, dtype=torch.uint8)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            actual = augment(images, "randaugment_cutout", torch.zeros(3), torch.ones(3), torch.Generator().manual_seed(2), torch.Generator().manual_seed(3))
        first = actual[0]
        independent = any(bool(((first != image) & (first != 0) & (image != 0)).any()) for image in actual[1:])
        self.assertTrue(independent)


if __name__ == "__main__":
    unittest.main()
