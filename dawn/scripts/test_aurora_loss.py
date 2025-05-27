import unittest
import torch
import aurora_loss


class myclass:
    pass


class LossTests(unittest.TestCase):
    def test_zeros(self):

        predictions = myclass()
        predictions.surf_vars = {
            "2t": torch.ones((1, 1, 720, 1440)),
            "10u": torch.ones((1, 1, 720, 1440)),
            "10v": torch.ones((1, 1, 720, 1440)),
            "msl": torch.ones((1, 1, 720, 1440)),
        }
        predictions.atmos_vars = {
            "t": torch.ones((1, 1, 13, 720, 1440)),
            "u": torch.ones((1, 1, 13, 720, 1440)),
            "v": torch.ones((1, 1, 13, 720, 1440)),
            "q": torch.ones((1, 1, 13, 720, 1440)),
            "z": torch.ones((1, 1, 13, 720, 1440)),
        }

        ground_truth = myclass()
        ground_truth.surf_vars = {
            "2t": torch.zeros((1, 1, 720, 1440)),
            "10u": torch.zeros((1, 1, 720, 1440)),
            "10v": torch.zeros((1, 1, 720, 1440)),
            "msl": torch.zeros((1, 1, 720, 1440)),
        }
        ground_truth.atmos_vars = {
            "t": torch.zeros((1, 1, 13, 720, 1440)),
            "u": torch.zeros((1, 1, 13, 720, 1440)),
            "v": torch.zeros((1, 1, 13, 720, 1440)),
            "q": torch.zeros((1, 1, 13, 720, 1440)),
            "z": torch.zeros((1, 1, 13, 720, 1440)),
        }

        loss = aurora_loss.mae(predictions, ground_truth)
        # self.assertLess(abs(loss - 1.8294444444444444), 0.0001)
        self.assertAlmostEqual(loss.item(), 1.8294444444444444, 6)

        loss = aurora_loss.mae(ground_truth, predictions)
        self.assertAlmostEqual(loss.item(), 1.8294444444444444, 6)

    def test_ones(self):

        predictions = myclass()
        predictions.surf_vars = {
            "2t": torch.ones((1, 1, 720, 1440)),
            "10u": torch.ones((1, 1, 720, 1440)),
            "10v": torch.ones((1, 1, 720, 1440)),
            "msl": torch.ones((1, 1, 720, 1440)),
        }
        predictions.atmos_vars = {
            "t": torch.ones((1, 1, 13, 720, 1440)),
            "u": torch.ones((1, 1, 13, 720, 1440)),
            "v": torch.ones((1, 1, 13, 720, 1440)),
            "q": torch.ones((1, 1, 13, 720, 1440)),
            "z": torch.ones((1, 1, 13, 720, 1440)),
        }

        loss = aurora_loss.mae(predictions, predictions)
        self.assertEqual(0, loss.item())

    def test_ones_721(self):

        predictions = myclass()
        predictions.surf_vars = {
            "2t": torch.ones((1, 1, 720, 1440)),
            "10u": torch.ones((1, 1, 720, 1440)),
            "10v": torch.ones((1, 1, 720, 1440)),
            "msl": torch.ones((1, 1, 720, 1440)),
        }
        predictions.atmos_vars = {
            "t": torch.ones((1, 1, 13, 720, 1440)),
            "u": torch.ones((1, 1, 13, 720, 1440)),
            "v": torch.ones((1, 1, 13, 720, 1440)),
            "q": torch.ones((1, 1, 13, 720, 1440)),
            "z": torch.ones((1, 1, 13, 720, 1440)),
        }

        ground_truth = myclass()
        ground_truth.surf_vars = {
            "2t": torch.ones((1, 1, 721, 1440)),
            "10u": torch.ones((1, 1, 721, 1440)),
            "10v": torch.ones((1, 1, 721, 1440)),
            "msl": torch.ones((1, 1, 721, 1440)),
        }
        ground_truth.atmos_vars = {
            "t": torch.ones((1, 1, 13, 721, 1440)),
            "u": torch.ones((1, 1, 13, 721, 1440)),
            "v": torch.ones((1, 1, 13, 721, 1440)),
            "q": torch.ones((1, 1, 13, 721, 1440)),
            "z": torch.ones((1, 1, 13, 721, 1440)),
        }

        loss = aurora_loss.mae(predictions, ground_truth)
        # self.assertLess(abs(loss - 1.8294444444444444), 0.0001)
        self.assertEqual(0, loss.item())

        with self.assertRaises(RuntimeError):
            aurora_loss.mae(ground_truth, predictions)


if __name__ == "__main__":
    unittest.main()
