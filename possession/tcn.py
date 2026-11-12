"""A temporal convolutional network for the two per-frame calls (M11l), applied with numpy alone.

The trees weigh each frame from its own row of features and centred means over 2.5 to 5 s. The
network reads the same design columns over about ten seconds either side of the frame and learns
its own filters over time: how the nearest players, the ball and the teams' shapes move into and
out of a possession. M11l tries it alone and averaged with the trees (possession.models.scores),
against the frozen pipeline, by the adoption rule of scripts/train_models.py.

The architecture is fixed before any run and never tuned on a held-out match: a 1x1 convolution
from the inputs to CHANNELS channels; then, for each dilation in DILATIONS, a residual block (ReLU,
a convolution of KERNEL taps spaced by the dilation, ReLU, a 1x1 convolution, added back); then
ReLU and a 1x1 convolution to two outputs, the log-odds that the ball is dead and, if it is alive,
that the home side has it: the two numbers the trees give. A frame's outputs can move with the
inputs reach() frames either side of it, 255 at 25 fps. Every layer pads its input with zeros
beyond the half, as PyTorch's padding does.

The inputs are the design columns standardised with the training frames' mean and scale, a
missing value set to 0, followed by one column per design column that is 1 where it was missing.

PyTorch fits the network on Kaggle (scripts/train_tcn.py), and the weights go into the model file
as lists. Here they are applied with numpy alone. Each output is summed bias first, then tap by tap
and input by input, every step an element-wise multiply and add. It is never a matrix product,
which BLAS may sum in a CPU-dependent order: the submission has to come out byte for byte on the
organisers' machine. ReLU and element-wise arithmetic are exact, so the same weights give the same
numbers on any CPU.
"""

import numpy as np

CHANNELS = 64
KERNEL = 3
DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128)


def reach(dilations=DILATIONS, kernel=KERNEL):
    """Frames on either side of a frame whose inputs can move its outputs."""
    return (kernel - 1) // 2 * sum(dilations)


def inputs(model, X):
    """(2 * columns, frames) float64: the design columns standardised with model['mean'] and
    model['scale'], a missing value as 0, then one row per column, 1 where it was missing."""
    X = np.asarray(X, dtype=np.float64)
    missing = np.isnan(X)
    standard = (X - np.asarray(model["mean"], dtype=np.float64)) \
        / np.asarray(model["scale"], dtype=np.float64)
    return np.ascontiguousarray(np.concatenate(
        [np.where(missing, 0.0, standard).T, missing.T.astype(np.float64)]))


def shifted(x, shift):
    """x moved along time so that column t holds x[:, t + shift], zeros beyond the half."""
    out = np.zeros_like(x)
    n = x.shape[1]
    if 0 <= shift < n:
        out[:, :n - shift] = x[:, shift:]
    elif -n < shift < 0:
        out[:, -shift:] = x[:, :n + shift]
    return out


def convolve(x, layer, dilation=1):
    """One convolution of x (channels, frames): out[o, t] = bias[o] + the sum over taps k and
    inputs i of weight[o][i][k] * x[i, t + (k - (taps - 1) // 2) * dilation], zeros beyond the
    half, as PyTorch's Conv1d with that padding computes it.

    The sum runs bias first, then tap by tap and input by input in that order, each step an
    element-wise multiply and add into the same array: the same numbers on any CPU.
    """
    weight = np.asarray(layer["weight"], dtype=np.float64)
    bias = np.asarray(layer["bias"], dtype=np.float64)
    outputs, channels, taps = weight.shape
    if channels != x.shape[0]:
        raise ValueError("a layer of {} inputs given {} channels".format(channels, x.shape[0]))
    out = np.empty((outputs, x.shape[1]), dtype=np.float64)
    out[:] = bias[:, None]
    product = np.empty_like(out)
    for k in range(taps):
        moved = shifted(x, (k - (taps - 1) // 2) * dilation)
        for i in range(channels):
            np.multiply(weight[:, i, k][:, None], moved[i][None, :], out=product)
            np.add(out, product, out=out)
    return out


def relu(x):
    return np.maximum(x, 0.0)


def scores(model, X):
    """(dead, home): the network's log-odds for every frame of one half, float64, as
    possession.models.score gives a tree model's."""
    h = convolve(inputs(model, X), model["input"])
    for block in model["blocks"]:
        r = convolve(relu(h), block["dilated"], block["dilation"])
        r = convolve(relu(r), block["pointwise"])
        h = h + r
    out = convolve(relu(h), model["output"])
    return out[0].copy(), out[1].copy()
