#holds all the different normalization and filtering methods

from scipy.stats.mstats import rankdata, mquantiles
from savitzky_golay import *
from numpy import mean

"""
A normalization function for comparing hotspots. Signals are mapped to a reference
distribution so that all hotspots, regardless of amplitude, can be compared.
Input:
v - the signal to be transformed
r - a reference distribution that v will be mapped onto
Output:
t - a transformation of v that preserves the rank of its values but mapped to r
"""
def normalize_quantilemap(v,r):
    ranks = rankdata(v)/len(v)
    t = mquantiles(r,prob=ranks)
    return t


"""
Returns a normalized and filtered vector appropriate for posterior decoding.
This method implements the first version, which divides the whole vector
by the mean of all counts greater than 4 and then applies a SG-filter,
2nd order 5 bp window 1st derivative.
Input:
v - the raw signal from a hotspot
Output:
v3 - the normalized and filtered first derivative signal
"""
def normalize_filter_threshold4(v):
    v2 = v/mean(v[v>4])
    v3 = savitzky_golay(v2,5,2,1,1)
    return v3

"""
Returns a normalized and filtered vector appropriate for posterior decoding.
This method implements the second version, which applies a SG-filter,
2nd order 5 bp window 1st derivative, and then divides by the absolute max.
Input:
v - the raw signal from a hotspot
Output:
v3 - the normalized and filtered first derivative signal
"""
def normalize_filter_maxderdivide(v):
    v2 = savitzky_golay(v,5,2,1,1)
    v3 = v2/max(abs(v2))
    return v3
