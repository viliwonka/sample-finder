import pandas as pd
import numpy as np
import functions

def generate_tests(signal):

    #always keep same seed so that results are consistent
    np.random.seed(1337)

    df = pd.DataFrame()

    df['ORIGINAL'] = signal
    df['ORIGINAL_INVERTED'] = -signal

    for i in range(3):
        df['GAUSS_5%_' + str(i)] = functions.add_gausian_noise(signal, 0.05)

    for i in range(3):
        df['GAUSS_10%_' + str(i)] = functions.add_gausian_noise(signal, 0.1)

    for i in range(3):
        df['GAUSS_25%_' + str(i)] = functions.add_gausian_noise(signal, 0.25)

    return df

def generate_tests_from_sample(sample, repetitions=40, gaps=30):
    
    #always keep same seed so that results are consistent
    np.random.seed(1337)
    N = len(sample)
    #permute repetitions and gaps
    p = np.random.permutation (([1]*repetitions) + ([0]*gaps))

    # start zero pad
    r = [pd.Series(np.zeros(N))]

    # repetitions of sample and zero gaps
    for n in p:
        if n == 1:
            r += [sample.copy()*(0.1 + 0.9*np.random.rand())]
        else:
            r += [pd.Series(np.zeros(N))]

    # final zero pad
    r += [pd.Series(np.zeros(N))]
   
    # construct a signal (named sample)
    sample = pd.concat(r, ignore_index=True)
    
    df = pd.DataFrame()

    df['REPEATED'] =  sample
    df['REPEATED_INVERTED'] = -sample

    # repeated with gauss 10% noise
    for i in range(3):
        df['RPTD_GAUSS_10%_' + str(i)] = functions.add_gausian_noise(sample, 0.10)

    # repeated with gauss 25% noise
    for i in range(3):
        df['RPTD_GAUSS_25%_' + str(i)] = functions.add_gausian_noise(sample, 0.25)

    # repeated with gauss 33% noise
    for i in range(3):
        df['RPTD_GAUSS_33%_' + str(i)] = functions.add_gausian_noise(sample, 0.33)

    # repeated with zero 10% noise
    for i in range(3):
        df['RPTD_ZERO_10%_' + str(i)] = functions.zero_noise(sample, 0.1)

    # repeated with zero 20% noise
    for i in range(3):
        df['RPTD_ZERO_20%_' + str(i)] = functions.zero_noise(sample, 0.2)

    # repeated with zero 20% noise
    for i in range(3):
        df['RPTD_ZERO_20%_' + str(i)] = functions.zero_noise(sample, 0.2)

    return df

def calculate_loss(original, noises):

    pass