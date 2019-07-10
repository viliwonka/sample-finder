import numpy as np
import pandas as pd
import scipy.signal
import functions
import operations

# imports csv file
# fixes notation (scientific numbers have ',' instead of '.')
def import_csv(file_name):
    # import file
    df = pd.read_csv(file_name, sep=';')
    # fix notation
    df['Y'] = df['Y'].apply(lambda x: x.replace(',', '.'))
    df['Y'] = df['Y'].astype(float)

    return df

# extracts non NaN numbers
# makes the window odd-numbered
# standardize it
def extract_original_signal(df):
        # get X with Y, where X is not NaN
    orig_sig = df[df == df]
    #removing one sample from X to make the orig_sig odd length, makes sliding window operations easier
    orig_sig = orig_sig[1:]
    orig_sig = orig_sig.reset_index(drop=True)
    # standardizing it
    orig_sig = functions.standardize(orig_sig)

    return orig_sig
