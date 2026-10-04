# Signal processing and machine learning

Ali Yaghi — academic signal-regression work and reproducible synthetic benchmarks.

## Tested entry point

```bash
python -m pip install -r requirements-demo.txt
python signal_pipeline.py
python signal_pipeline.py --neural --output neural_metrics.json
python -m unittest -v test_signal_pipeline
```

`signal_pipeline.py` generates independent noisy sinusoid datasets (500 training, 200 test, seed 42). It estimates **amplitude in arbitrary units**, using RMS, mean absolute value and FFT amplitude features. Scaling is fitted on training data only. A Ridge model and an optional MLP are compared with a conventional FFT baseline. The FFT baseline is slightly better in this example; the code does not imply an AI advantage or industrial performance.

## Original research code

The historical `src/` directory is retained for provenance. It contains TensorFlow experiments on interferometric signals. It is **not the supported entry point**: local paths and assumptions about experimental file/channel formats require adaptation. In particular, `src/evaluate.py` evaluates the configured dataset rather than an independent held-out dataset and does not apply the fitted training scaler. Do not use its output as an independent test score.

For experimental work, record channel semantics explicitly, split by acquisition/trajectory **before** overlapping windows, and save the fitted preprocessing together with the model. No experimental datasets or trained laboratory models are redistributed here.

The standalone benchmark above is a new, clearly labelled reproducible addition, not a replacement for validation on the original displacement experiment.

See [VALIDATION.md](VALIDATION.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

Related projects: [scientific portfolio](https://github.com/yaghi-ali/scientific-projects). Contact: contact@optiia-consulting.fr.
