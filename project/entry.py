import numpy as np
import pandas as pd
import scipy.signal
import functions
import operations
import importing
import counter
import tests
import plotting

# run tests with noise
def run_tests(sample, signal):

    # counter with default hypervalues
    default_counter = counter.Counter(sample)

    # signal created from sample, copied multiple times and distrorted    
    df_s = tests.generate_tests_from_sample(sample)

    # signal, original and distorted multiple times
    df = tests.generate_tests(signal)

    def show_result(name, counter, signal, do_print=False, plot=True):   
        count_result = counter.count(signal)
        format_str = "{}:\nunified_count: {}, sub_counts:{}\nnn_count:{}, nn_sub_count:{}\n"
        to_print = format_str.format(name, count_result['unified_count'], count_result['max_counts'], count_result['unified_marked_count'], count_result['max_marked_counts'])       
        print(to_print)

        if plot==True: 
            plotting.plot_extrema_df_marked(signal, name, count_result)

    print("REAL DATA")
    for name in df:
        show_result(name, default_counter, df[name], True, True)

    print("SYNTHETIC DATA")
    #
    for name in df_s:
        show_result(name, default_counter, df_s[name], True, True)
    
    #for i in range(len(gauss_10_noise)):
    #    print_result(str(i)+" gauss noise 10%%", default_counter, gauss_10_noise[i])
    
    #for i in range(len(gauss_25_noise)):
    #    print_result(str(i)+" gauss noise 25%%", default_counter, gauss_25_noise[i])

# stress test
def main():

    print("Starting main..\n")

    df = importing.import_csv('test_signals.csv')
    orig_sig = importing.extract_original_signal(df['X'])

    print("Data loaded\n")

    print("Running tests:")
    run_tests(orig_sig, df['Y'])

if __name__ == "__main__":
    main()
