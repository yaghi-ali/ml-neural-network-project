# Validation — 2026-10-04

`python -m unittest -v test_signal_pipeline`: **1 test passed**, verifying training-only scaling and the deterministic Ridge benchmark.

Independent synthetic datasets, seed 42, 500 training and 200 test samples:

| Method | Test amplitude MAE (arbitrary units) |
|---|---:|
| FFT baseline | 0.0130456684 |
| Ridge | 0.0145246890 |
| MLP (32,16) | 0.0292757413 |

The simple FFT baseline wins in this synthetic fixed-frequency example. No real displacement accuracy is inferred. The historical TensorFlow scripts were not executed or validated. Tested with scikit-learn 1.9.1 and NumPy 2.5.3 on Windows/Python 3.12.
