import functions
import operations
import struct
import pandas as pd
import numpy as np

####################
# TODO LATER MAYBE
class Processor():

    def __init__(self):

        self.window_length = 9
        self.hyper_params = {}
        self.neki = None
####################

# class made for counting the signal
class Counter():

    # this initialization includes lots of hyper-parameters
    def __init__(self, sample, hyper_params = None):

        if hyper_params == None:
            hyper_params = {}

        ### initialize default parameters, if none
        def defaultize(param_name, default):
            if param_name not in hyper_params:
                hyper_params[param_name] = default

        # what processors to use (SampleOperator methods)
        defaultize('processors', ['MIN_L2_THRS', 'MIN_L1_THRS', 'CORR'])
        # window_length of window, that will be used for sliding max operation
        defaultize('window_length', 7)
        # min_alpha for L2_THRS
        defaultize('min_alpha', 0.2)
        # max_alpha for L2_THRS
        defaultize('max_alpha', 10)
        # threshold used for counting maximums
        defaultize('max_threshold', 0.17)
        # method used for calculating delta of maximums (possible choices are 'min', 'avg', 'harm', 'geo')
        defaultize('delta_method', 'geo')
        
        # initialize members
        self.sample        = sample
        self.hyper_params  = hyper_params
        self.op            = operations.SampleOperator(sample)
        self.w_op          = operations.WindowOperator(hyper_params['window_length'])
        self.processors    = hyper_params['processors']
        self.min_alpha     = hyper_params['min_alpha']
        self.max_alpha     = hyper_params['max_alpha']
        self.max_threshold = hyper_params['max_threshold'] 
        self.delta_method  = hyper_params['delta_method']

    # counts the occurence of sample in signal
    def count(self, signal):

        # PMFs
        pmfs = pd.DataFrame()

        # normal MIN L2
        if 'MIN_L2' in self.processors:
            pmfs['MIN_L2_MAX']      = functions.convert_to_PMF( self.w_op.apply_max( functions.rbf( self.op.apply_min_error_L2( signal))))

        # thresholded MIN L2
        if 'MIN_L2_THRS' in self.processors:
            pmfs['MIN_L2_THRS_MAX'] = functions.convert_to_PMF( self.w_op.apply_max( functions.rbf( self.op.apply_min_error_L2_thresholded( signal, self.min_alpha, self.max_alpha))))

       # normal MIN L1
        if 'MIN_L1' in self.processors:
            pmfs['MIN_L1_MAX']      = functions.convert_to_PMF( self.w_op.apply_max( functions.rbf( self.op.apply_min_error_L1( signal))))

        # thresholded MIN L1
        if 'MIN_L1_THRS' in self.processors:
            pmfs['MIN_L1_THRS_MAX'] = functions.convert_to_PMF( self.w_op.apply_max( functions.rbf( self.op.apply_min_error_L1_thresholded( signal, self.min_alpha, self.max_alpha))))

        # correlation
        if 'CORR' in self.processors:
            pmfs['CORR_MAX']        = functions.convert_to_PMF( self.w_op.apply_max( self.op.apply_cross_correlate( signal).abs()))
       
        extremas = pd.DataFrame()
        max_counts = {}
        max_locations = {}
        
        # for each processed signal, find maximums        
        # pmf_name is string
        for pmf_name in pmfs:
            
            extremas[pmf_name] = functions.find_extremas(pmfs[pmf_name])
            m = functions.count_maximums(pmfs[pmf_name], extremas[pmf_name], self.max_threshold, self.delta_method)
            max_counts[pmf_name] = m[0]
            max_locations[pmf_name] = m[1]

        # find nn extremas and mark them
        functions.find_nn_extrema(max_locations)

        max_marked_counts = {}
        max_marked_locations = {}
        # now count how many have been marked
        for pmf_name in max_locations:
            extrema = max_locations[pmf_name]
            count = 0
            max_marked_locations[pmf_name] = []
            for e in extrema:
                if e.marked == True:
                    count +=1
                    max_marked_locations[pmf_name] += [e]

            max_marked_counts[pmf_name] = count

        max_counts_values = list(max_counts.values())
        max_counts_marked_values = list(max_marked_counts.values())
        # calculate unified count, should change it for something better, or add more processor functions
        # maybe could try avg. mean, geometric mean or harmonic mean, too
        unified_count = np.median(max_counts_values)
        unified_marked_count = np.median(max_counts_marked_values)
        # return complete set of all items used in calculation, useful for benchmarking and plotting

        d = {

            'unified_count' : unified_count,
            'unified_marked_count': unified_marked_count,
            'max_counts' : max_counts,
            'max_marked_counts': max_marked_counts,
            'max_locations': max_locations,
            'max_marked_locations': max_marked_locations,
            'pmfs' : pmfs,
            'extremas' : extremas
        }

        return d