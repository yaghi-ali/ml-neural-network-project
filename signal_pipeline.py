"""Reproducible signal regression demo; synthetic amplitude, not displacement."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error


def dataset(n, rng):
    t = np.arange(256)/256
    amplitude = rng.uniform(.2,2,n)
    phase = rng.uniform(0,2*np.pi,n)
    noise = rng.uniform(.02,.35,n)
    signals = amplitude[:,None]*np.sin(2*np.pi*16*t+phase[:,None])+rng.normal(size=(n,256))*noise[:,None]
    fft = 2*np.abs(np.fft.rfft(signals,axis=1)[:,16])/256
    features = np.column_stack([np.sqrt(np.mean(signals**2,axis=1)),np.mean(np.abs(signals),axis=1),fft])
    return features,amplitude,fft


def run(seed=42, neural=False):
    rng = np.random.default_rng(seed)
    train_x,train_y,_ = dataset(500,rng)
    test_x,test_y,baseline = dataset(200,rng)
    estimator = (MLPRegressor(hidden_layer_sizes=(32,16),max_iter=1000,early_stopping=True,
                             random_state=seed) if neural else Ridge(alpha=1))
    model = make_pipeline(StandardScaler(),estimator)
    model.fit(train_x,train_y)
    predicted = model.predict(test_x)
    metrics = dict(seed=seed,train_samples=500,test_samples=200,
                   model='MLP' if neural else 'Ridge',
                   model_mae=float(mean_absolute_error(test_y,predicted)),
                   fft_mae=float(mean_absolute_error(test_y,baseline)),
                   data='synthetic fixed-frequency sinusoids; amplitude in arbitrary units')
    return model, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--neural',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('demo_metrics.json'))
    args=parser.parse_args()
    _,metrics=run(neural=args.neural)
    args.output.write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2))
