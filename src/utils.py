import numpy as np
import scipy as sp


def seq_len(text):
    return len(text.split(" "))

def softmax(vec):
    vec = vec - np.max(vec, axis=-1, keepdims=True)
    vec = np.exp(vec)
    sum_exp = np.sum(vec, axis=-1, keepdims=True)
    return vec / sum_exp

def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def ReLU(x):
    return np.maximum(0,x)

def logsumexp(x, axis=None, keepdims=False):
    x = np.asarray(x)
    m = np.max(x, axis=axis, keepdims=True)
    y = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return y if keepdims else np.squeeze(y, axis=axis)

def gaussian_ppf(p, mu=0.0, sigma=1.0):
    return mu + sigma * np.sqrt(2) * sp.special.erfinv(2*p - 1)

