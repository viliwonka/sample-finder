import scipy.signal
import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr
from functions import *

# Window operator has methods that only operate based on window length
class WindowOperator():

    def __init__(self, length): 
        self.length = length

    ###rms, requires sample just for sample_len
    def apply_rms(self, signal):

        def apply_func(window):
            return np.sqrt((window*window).mean())

        return signal.rolling(self.length, center=True).apply(apply_func, raw=True)

    ###mean
    def apply_mean(self, signal):
        return signal.rolling(self.length, center=True).mean()

    ###median
    def apply_median(self, signal):
        return signal.rolling(self.length, center=True).median()

    ###max
    def apply_max(self, signal):
        return signal.rolling(self.length, center=True).max()

    ###min
    def apply_min(self, signal):
        return signal.rolling(self.length, center=True).min()

    ###min&max, square kind of function
    def apply_min_max(self, signal):

        def apply_func(window):

            pos_min = window.argmin()
            pos_max = window.argmax()

            acc = 0

            if (pos_min != 0 and pos_min != (self.length - 1)):
                acc-=1

            if (pos_max != 0 and pos_max != (self.length - 1)):
                acc+=1
                
            return acc

        return signal.rolling(self.length, center = True).apply(apply_func, raw = True)

# SampleOperator contains methods that store internal Sample for operations
class SampleOperator(WindowOperator):
    
    def __init__(self, sample): 
        self.sample = sample
        self.length = len(sample)

    ### cross correlation
    def apply_cross_correlate(self, signal):

        def apply_func(window):
            return (self.sample*window).mean()

        return signal.rolling(self.length, center=True).apply(apply_func, raw=True)

    ### min error L2
    def apply_min_error_L2(self, signal):

        def apply_func(window):
            return min_errorL2(self.sample, window)[0]

        return signal.rolling(self.length, center=True).apply(apply_func, raw=True)

    ### min error L2
    def apply_min_error_L2_thresholded(self, signal, min_alpha = 0.05, max_alpha=50.0):
        
        def apply_func(window):
            return min_errorL2_thresholded(self.sample, window, min_alpha, max_alpha)[0]

        return signal.rolling(self.length, center=True).apply(apply_func, raw=True)

    ### min error L1
    def apply_min_error_L1(self, signal):

        def apply_func(window):
            return min_errorL1(self.sample, window)[0]

        return signal.rolling(self.length, center=True).apply(apply_func, raw=True)

    ### min error L1
    def apply_min_error_L1_thresholded(self, signal, min_alpha = 0.05, max_alpha=50.0):
        
        def apply_func(window):
            return min_errorL1_thresholded(self.sample, window, min_alpha, max_alpha)[0]

        return signal.rolling(self.length, center=True).apply(apply_func, raw=True)

    ### pearson
    def apply_pearson(self, signal):

        def pearson(window):
            result = pearsonr(self.sample, window)
            # result contains 
            return result[0]

        return signal.rolling(self.length, center=True).apply(pearson, raw=True)
