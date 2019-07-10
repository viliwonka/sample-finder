import matplotlib.pyplot as plt
import matplotlib.collections as collections
import functions
import numpy as np

def plot_extrema(pmf, extrema):

    e_plot = extrema[0]

    fig, ax = plt.subplots(figsize=(40,12))

    ax.plot(pmf.index, pmf, color='black')
    ax.axhline(0, color='black', lw=2)

    collection = collections.BrokenBarHCollection.span_where(
        e_plot.index, ymin=0, ymax=1, where= e_plot > 0, facecolor='green', alpha=0.5, linewidths=1, edgecolors='green')
    ax.add_collection(collection)

    collection = collections.BrokenBarHCollection.span_where(
        e_plot.index, ymin=0, ymax=1, where= e_plot < 0, facecolor='red', alpha=0.5, linewidths=1,edgecolors='red')
    ax.add_collection(collection)

    plt.show()

def plot_extrema_df(signal, signal_name, pmf_df, extrema_df):

    #signal plot plus pmf/extrema
    fig, axs = plt.subplots(1+len(pmf_df.columns), 1, figsize=(40,9))
    
    axs[0].plot(signal.index, signal, color='black') 
    axs[0].set_title(signal_name)

    i = 1
    for name in pmf_df:

        e_plot = extrema_df[name][0]
        
        axs[i].plot(pmf_df[name].index, pmf_df[name], color='black')
        #axs[i].axhline(0, color='black', lw=2)

        collection = collections.BrokenBarHCollection.span_where(
            e_plot.index, ymin=0, ymax=1, where= e_plot > 0, facecolor='green', alpha=0.5, linewidths=1, edgecolors='green')
        axs[i].add_collection(collection)

        collection = collections.BrokenBarHCollection.span_where(
            e_plot.index, ymin=0, ymax=1, where= e_plot < 0, facecolor='red', alpha=0.5, linewidths=1,edgecolors='red')
        axs[i].add_collection(collection)
        
        axs[i].set_title(name)
        
        i += 1

    axs[i-1].set_xlabel('time')

    for ax in axs.flat:
        ax.label_outer()

    plt.show()

# extrema_df_marked
def plot_extrema_df_marked(signal, signal_name, params_dict):

    pmf_df = params_dict['pmfs']
    extrema_df = params_dict['extremas']
    selected_extrema_df = params_dict['max_locations']
    selected_extrema_nn_df = params_dict['max_marked_locations']

    #signal plot plus pmf/extrema
    fig, axs = plt.subplots(1+len(pmf_df.columns), 1, figsize=(40,9))
    
    axs[0].plot(signal.index, signal, color='black') 
    axs[0].set_title(signal_name)
    
    i = 1
    for name in pmf_df:

        e_plot = extrema_df[name][0]
        
        pmf_df[np.isnan(pmf_df)] = 0.0

        axs[i].plot(pmf_df[name].index, pmf_df[name], color='black')
        #axs[i].axhline(0, color='black', lw=2)

        collection = collections.BrokenBarHCollection.span_where(
            e_plot.index, ymin=0, ymax=1, where= e_plot > 0, facecolor='green', alpha=0.15, linewidths=1, edgecolors='green')
        axs[i].add_collection(collection)

        # special maximums, that were selected
        mark = functions.mark_series_extrema(np.zeros(len(signal)),  selected_extrema_df[name])
        mark_nn = functions.mark_series_extrema(np.zeros(len(signal)),  selected_extrema_nn_df[name])

        #mark marked maximums
        collection = collections.BrokenBarHCollection.span_where(
            e_plot.index, ymin=0, ymax=1, where= mark > 0, facecolor='yellow', alpha=0.4, linewidths=1, edgecolors='yellow')
        axs[i].add_collection(collection)

        #mark marked maximums with neighbors
        collection = collections.BrokenBarHCollection.span_where(
            e_plot.index, ymin=0, ymax=1, where= mark_nn > 0, facecolor='orange', alpha=0.5, linewidths=2, edgecolors='orange')
        axs[i].add_collection(collection)

        # mark minimums (red)
        collection = collections.BrokenBarHCollection.span_where(
            e_plot.index, ymin=0, ymax=1, where= e_plot < 0, facecolor='red', alpha=0.15, linewidths=1,edgecolors='red')
        axs[i].add_collection(collection)
        
        axs[i].set_title(name)
        
        i += 1

    axs[i-1].set_xlabel('time')

    for ax in axs.flat:
        ax.label_outer()

    plt.show()

    pass