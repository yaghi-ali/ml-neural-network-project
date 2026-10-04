import unittest
import numpy as np
from signal_pipeline import dataset, run


class PipelineChecks(unittest.TestCase):
    def test_scaling_uses_training_data_only(self):
        model, metrics = run()
        x, _, _ = dataset(500,np.random.default_rng(42))
        np.testing.assert_allclose(model[0].mean_,x.mean(axis=0))
        self.assertLess(metrics['model_mae'], .05)
        self.assertAlmostEqual(metrics['model_mae'],.014524689,places=7)
        self.assertLess(metrics['fft_mae'],metrics['model_mae'])


if __name__ == '__main__': unittest.main()
