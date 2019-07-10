import scipy.signal
import pandas as pd
import numpy as np
from structs import *

def standardize(x):
    return (x - x.mean()) / x.std()
 
# radius basis function
epsilon = 0.00001

def sign(x):
    x = x.copy()
    x[np.isnan(x)] = 0.0
    return np.sign(x)

# radial basis function
def rbf(r): 
    
    s = sign(r)
    r = np.abs(r)
    return s / (1.0 + r*r)

# radial square root
def rsq(r): 
    
    s = np.sign(r)
    r = np.abs(r)
    return s * np.sqrt(r)

# returns as minimal as possible L2 norm 
def min_errorL2(x, y):
    
    mean_factor = 1.0 / len(x)
    # x and y are vectors of same length
    # alpha is projection length, solution to min(theta) ||x-theta*y||
    # alpha minimizes norm
    alpha = (x*y).sum() / ( (y*y).sum() + epsilon)
    
    # difference
    r = (x - (alpha * y))
    
    # return the calculated L2 norm and alpha
    return (np.sqrt(mean_factor * (r*r).sum()), alpha)
 
# returns approximation of minimal as possible L1 norm 
def min_errorL1(x, y):
    
    mean_factor = 1.0 / len(x)
    # modified version of min_errorL2
    # x and y are vectors of same length
    # alpha minimizes norm
    alpha = (x*y).sum() / ( (y*y).sum() + epsilon)
    
    # difference
    r = (x - (alpha * y))
    
    # return the calculated L1 norm and alpha
    return (mean_factor * (np.abs(r).sum()), alpha)

# returns similar to min_errorL2, but with clamped alpha (so no absurd scaling occurs)
def min_errorL2_thresholded(x, y, min_alpha=None, max_alpha=None):
    
    if (min_alpha != None and min_alpha <= 0.0) or (max_alpha != None and max_alpha <= 0.0) or (min_alpha != None and max_alpha != None and min_alpha >= max_alpha):

        raise Exception("min_alpha={} or max_alpha={} are invalid".format(min_alpha, max_alpha))

    mean_factor = 1.0 / len(x)
    # x and y are vectors of same length
    # alpha is projection length, solution to min(theta) ||x-theta*y||
    # alpha minimizes norm
    alpha = (x*y).sum() / ( (y*y).sum() + epsilon)
    
    #if absolute alpha is more than max_alpha, clamp it! 
    if max_alpha != None and max_alpha < np.abs(alpha):
        alpha = max_alpha * np.sign(alpha)

    # if absolute alpha is less than min_alpha, clamp it, but more important
    if min_alpha != None and min_alpha > np.abs(alpha):       
        alpha = min_alpha * np.sign(alpha)
        
    # difference
    r = (x - (alpha * y))
    
    # return the calculated L2 norm and alpha
    return (np.sqrt(mean_factor * (r*r).sum()), alpha)
 
# returns similar to min_errorL2, but with clamped alpha (so no absurd scaling occurs)
def min_errorL1_thresholded(x, y, min_alpha=None, max_alpha=None):
    
    if (min_alpha != None and min_alpha <= 0.0) or (max_alpha != None and max_alpha <= 0.0) or (min_alpha != None and max_alpha != None and min_alpha >= max_alpha):

        raise Exception("min_alpha={} or max_alpha={} are invalid".format(min_alpha, max_alpha))

    mean_factor = 1.0 / len(x)
    # x and y are vectors of same length
    # alpha is projection length, solution to min(theta) ||x-theta*y||
    # alpha minimizes norm
    alpha = (x*y).sum() / ( (y*y).sum() + epsilon)
    
    #if absolute alpha is more than max_alpha, clamp it! 
    if max_alpha != None and max_alpha < np.abs(alpha):
        alpha = max_alpha * np.sign(alpha)

    # if absolute alpha is less than min_alpha, clamp it, but more important
    if min_alpha != None and min_alpha > np.abs(alpha):       
        alpha = min_alpha * np.sign(alpha)
        
    # difference
    r = (x - (alpha * y))
    
    # return the calculated L1 norm and alpha
    return (mean_factor * (np.abs(r).sum()), alpha)
 
# remove minimal signals based on threshold
def remove_noise(signal, threshold = epsilon):
    
    denoised = signal.copy()
    denoised[denoised.abs() < threshold] = 0.0

    return denoised

# normalize vector to probability mass function
def convert_to_PMF(signal):

    copy = signal.copy()
    copy -= np.min(signal)
    copy /= np.sum(signal)

    return copy

# finds maximums and minimums
def find_extremas(signal):
    
    N = len(signal) 
    groups = []
    prev_index = 0

    for i in range(1, N):
        
        s_0 = signal[i-1]
        s_1 = signal[i  ]
        
        if s_0 == s_1:
            continue
        else:
            # each group has mean, left interval, right interval
            groups += [(signal[prev_index], prev_index, i)]
            prev_index = i

    minima_extrema = []
    maxima_extrema = []
    all_extrema = []

    prev_extrema = None

    # consists of -1, 0 and +1
    trail = pd.Series(np.zeros(N))

    # compare the minimums
    for i in range(1, len(groups)-1):
        
        s_0 = groups[i-1][0]
        s_1 = groups[i  ][0]
        s_2 = groups[i+1][0]
        
        # local minimum
        if s_0 > s_1 < s_2:

            g = groups[i]
            e = Extrema('min', (g[1], g[2]), g[0], prev_extrema)
            trail[g[1]:g[2]] = -1
 
            if prev_extrema != None:
                prev_extrema.right_neighbour = e 

            minima_extrema += [e]
            all_extrema    += [e]
            prev_extrema = e

        # local maximum
        elif s_0 < s_1 > s_2:

            g = groups[i]
            e = Extrema('max', (g[1], g[2]), g[0], prev_extrema)
            trail[g[1]:g[2]] = +1

            if prev_extrema != None:
                prev_extrema.right_neighbour = e 

            maxima_extrema += [e]
            all_extrema    += [e]
            prev_extrema = e
    
    return (trail, minima_extrema, maxima_extrema, all_extrema)

# count extremas based on treshold
def count_maximums(pmf, extrema, threshold = 0.1, delta_method='min'):

    # get max number of pmf
    m = pmf.max()

    # index 2 = maxima
    maxima = extrema[2]
    count = 0

    maximums = []
    
    for i in range(len(maxima)):

        e = maxima[i]
        d = None

        # min & avg
        if delta_method == 'min':
            d = e.min_delta() 
        elif delta_method == 'avg':
            d = e.avg_delta()
        elif delta_method == 'harm':
            d = e.h_delta()
        elif delta_method == 'geo':
            d = e.g_delta()

        # apply normalizing factor
        d *= (1 / m)

        if d > threshold:
            count += 1
            maximums += [e]

    return count, maximums

# generates noise and adds it onto signal
def add_gausian_noise(signal, std_level):

    copy = signal.copy()
    std_of_signal = copy.std()
    # gaussian noise
    noise = np.random.normal(scale = std_of_signal * std_level, size=len(signal))

    copy += noise

    return copy
 
# add zeros to random spots
def zero_noise(signal, percent):

    copy = signal.copy()

    r = np.random.rand(len(copy)) 
    # zero random elements
    copy[r < percent] = 0.0
    return copy

# mark series with 1, where there is interval (extrema)
def mark_series_extrema(series, extrema):

    for m in extrema:
        series[m.index[0]:m.index[1]] = 1

    return series

# find nearest neighbour extrema
def find_nn_extrema(extrema_dict):

    keys = list(extrema_dict.keys())
    # i is offset
    i = 1
    # triangular iteration, kind of
    for k0 in keys:
        e0_list = extrema_dict[k0]
        for k1 in keys[i:]:
            e1_list = extrema_dict[k1]
            # now compare list with list, this part could be MUCH MORE optimized (binary tree), I just don't have time right now to do it
            for e0 in e0_list:
                for e1 in e1_list:
                    # test for overlap then mark the extrema
                    if e0.does_overlap(e1):
                        e0.marked = True
                        e1.marked = True
        #remove this i and you break the algo!
        i+=1

