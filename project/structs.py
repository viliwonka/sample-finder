import scipy.signal
import pandas as pd
import numpy as np
import functions

class Extrema:

    def __init__(self, extrema_type, index, value, left_neighbour=None, right_neighbour=None):

        if extrema_type != 'max' and extrema_type != 'min':
            raise Exception('Invalid extrema type')
  
        self.extrema_type    = extrema_type
        self.index           = index
        self.value           = value        
        self.left_neighbour  = left_neighbour
        self.right_neighbour = right_neighbour
        self.marked          = False
    # get absolute deltas 
    def _get_deltas(self):
        
        left = right = 0.0
        if self.left_neighbour != None:
            left = self.left_neighbour.value

        if self.right_neighbour != None:
            right = self.right_neighbour.value

        d0 = self.value - left
        d1 = self.value - right

        return (d0, d1)

    # returns average between two deltas        
    def avg_delta(self):
        (d0, d1) = self._get_deltas()       
        return (np.abs(d0) + np.abs(d1)) / 2.0
 
    # returns minimum delta
    def min_delta(self):
        (d0, d1) = self._get_deltas()
        return min(np.abs(d0), np.abs(d1))

    # returns harmonic mean of deltas
    def h_delta(self):
        (d0, d1) = self._get_deltas()
        # epsilon to avoid division by zero
        h0 = 1.0 / (d0 + functions.epsilon)
        h1 = 1.0 / (d1 + functions.epsilon)
        return 1.0 / ( h0 + h1 )

    # returns geometric mean of deltas
    def g_delta(self):
        (d0, d1) = self._get_deltas()
        return np.sqrt( d0 * d1 )

    # checks if interval overlap
    def does_overlap(self, other):
        return (other.index[0] <= self.index[0] <= other.index[1]) or (self.index[0] <= other.index[0] <= self.index[1])
