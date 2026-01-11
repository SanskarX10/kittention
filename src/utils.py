import numpy as np
import scipy as sp


def seq_len(text):
    return len(text.split(" "))

def softmax(vec):
    vec = vec - np.max(vec, axis=-1, keepdims=True)
    vec = np.exp(vec)
    sum_exp = np.sum(vec, axis=-1, keepdims=True)
    return vec / sum_exp


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def ReLU(x):
    return np.maximum(0,x)

def logsumexp(x, axis=None, keepdims=False):
    x = np.asarray(x)
    m = np.max(x, axis=axis, keepdims=True)
    y = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return y if keepdims else np.squeeze(y, axis=axis)


def l2_normalize(x, axis=-1, eps=1e-12):
    norm = np.sqrt(np.sum(x**2, axis=axis, keepdims=True) + eps)
    return x / norm

def gaussian_ppf(p, mu=0.0, sigma=1.0):
    return mu + sigma * np.sqrt(2) * sp.special.erfinv(2*p - 1)

def rms_norm(x, eps=1e-6):
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return x / rms

def swish(x):
    return x * sigmoid(x)

    
def short_conv1d(x, kernel_size=4):
    '''
    for kimideltaattenion
    Causal 1D convolution for local context.
    x: (seq_len, channels)
    Returns: (seq_len, channels)
    '''
    seq_len, channels = x.shape
    kernel = np.random.rand(kernel_size, channels, channels) / kernel_size
    
    output = np.zeros_like(x)
    for t in range(seq_len):
        for k in range(min(kernel_size, t + 1)):
            output[t] += x[t - k] @ kernel[k]
    return output

