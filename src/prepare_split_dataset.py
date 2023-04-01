import os
import nltk
import spacy
import pickle
import re
import torch
from torch.nn import functional as F
from transformers import BertTokenizerFast, BertTokenizer,RobertaTokenizer,T5TokenizerFast
import emojis
import demoji
import random
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import unicodedata
from textblob import TextBlob
import numpy as np
x1 = torch.Tensor([[1, 2, 3],[-10,4,7],[0,8,2],[-3,5,-4]])
x2 = torch.Tensor([[1, 5, 3],[-10,4,7],[0,-8,2],[3,10,7]])



def rescale(value, orig_min, orig_max, new_min, new_max):
    """
    Rescales a `value` in the old range defined by
    `orig_min` and `orig_max`, to the new range
    `new_min` and `new_max`. Assumes that
    `orig_min` <= value <= `orig_max`.
    Parameters
    ----------
    value: float, default=None
        The value to be rescaled.
    orig_min: float, default=None
        The minimum of the original range.
    orig_max: float, default=None
        The minimum of the original range.
    new_min: float, default=None
        The minimum of the new range.
    new_max: float, default=None
        The minimum of the new range.
    Returns
    ----------
    new_value: float
        The rescaled value.
    """

    orig_span = orig_max - orig_min
    new_span = new_max - new_min

    try:
        scaled_value = float(value - orig_min) / float(orig_span)
    except ZeroDivisionError:
        orig_span += 1e-6
        scaled_value = float(value - orig_min) / float(orig_span)

    return new_min + (scaled_value * new_span)


def correct_scores(scores, p1, p2):
    new_scores = []
    for sc in scores:
        if sc <= p1:
            sc = rescale(sc, 0, p1, 0, 0.49)
            new_scores.append(sc)
        elif sc > p1 and sc < p2:
            new_scores.append(0.5)
        else:
            sc = rescale(sc, p2, 1, 0.51, 1)
            new_scores.append(sc)
    return np.array(new_scores)

sim =F.cosine_similarity(x1,x2).cpu().data.numpy()

print(sim)

# cor = correct_scores(sim,0.6,0.6)
inv = 1 - sim
preds = (sim > 0.6) * 1
print(preds)
# cor = correct_scores(sim,0.4,0.6)
# print(cor)
def my_thresholder(sims, low_thresh, high_thresh, epsilon=1e-6, min_decision=-1, max_decision=-1):
    if min_decision == -1:
        min_decision = low_thresh - epsilon

    if max_decision == -1:
        max_decision = high_thresh + epsilon

    fixed_sims = sims.copy()
    # need to make sure this is < 0.5 - just normalize to [0, 0.5)

    fixed_sims[np.logical_and(sims < low_thresh, sims > 0.5)] = 1-fixed_sims[np.logical_and(sims < low_thresh, sims > 0.5)]

    # need to make sure samples above high thresh is normalized to be in (0.5, 1]

    fixed_sims[sims > high_thresh] = fixed_sims[sims > high_thresh]
    fixed_sims[np.logical_and(sims >= low_thresh, sims <= high_thresh)] = 0.5
    return fixed_sims

sim = np.array([ 0.9035079,1.0,-0.88235295,0.65626142])

new = my_thresholder((1 + sim) / 2,0.6,0.65)
# print(my_thresholder(sim,0.4,0.6))
print(new)


