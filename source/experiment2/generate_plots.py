#
# Copyright © 2022 Politecnico di Torino, Control and Computer Engineering Department, SMILIES group
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
# associated documentation files (the "Software"), to deal in the Software without restriction, 
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial 
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT 
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.cm as cm


# a function to plot the final total number of cells
# filepath - filepath of the output csv
# experiment - indication of the experiment
# exploration - indication of the hyperparameter exploration
def plotPerformance_final_cells(filepath, experiment, exploration):
    raw_data = pd.read_csv(filepath)
    data_T = raw_data.T
    data_T.columns = data_T.iloc[0]
    data_T = data_T[1:]

    if exploration == 'lr_gamma':
        data = data_T[['lr=0.001 gamma=0.95', 'lr=0.0001 gamma=0.95', 'lr=0.001 gamma=0.99', 'lr=0.0001 gamma=0.99']]
    elif exploration == 'numIter':
        data = data_T[['40 iterations/step', '50 iterations/step', '20 iterations/step']]
    else:
        print('Insert valid exploration name!')

    data.plot(figsize=(10, 5), fontsize=15, lw=4)
    plt.xlabel('Epoch', fontsize=15)
    if experiment == '1_final_n_cells':
        plt.ylabel('Final number of cells', fontsize=15)
    if experiment == '2_final_fraction_cells':
        plt.ylabel('Final fraction of cells inside target', fontsize=15)
    plt.legend(loc=3)
    plt.savefig(experiment + '_' + exploration + '.png')

# a function to plot performance metrics - either mean or variance
# filepathRaw - filepath of the output csv
# experiment - indication of the experiment
# exploration - indication of the hyperparameter exploration 
# metric - the metric to consider (mean or variance)
# logScale - provides logScale visualization on the y axis if True (defaults to False)
# combined - indicates whether twp epoch blocks must be combined (defaults to False)
# filepath2 - file path of the raw performance metrics of the second block of epochs (needed if combined=True)
def plotPerformance_metrics_lines(filepathRaw, experiment, exploration, metric, logScale=False, combined=False, filepath2=""):
    
    # combine data from the two epoch blocks if required
    if combined:
        dataRaw=concatEpochs(filepathRaw, filepath2)
    else:
        raw_dataRaw = pd.read_csv(filepathRaw)
        dataRaw_T = raw_dataRaw.T
        dataRaw_T.columns = dataRaw_T.iloc[0]
        dataRaw = dataRaw_T[1:]
    
    if logScale:
        #min and max over df
        min=dataRaw.min().min()
        max=dataRaw.max().max()
        #print(min,max)

        for c in dataRaw.columns:

            # min max normalization with min and max over column
            dataRaw[c] = (dataRaw[c] - dataRaw[c].min()) / (dataRaw[c].max() - dataRaw[c].min())

            # min max normalization with min and max over df
            dataRaw[c] = (dataRaw[c] - min) / (max - min)

            #log10 transformation
            dataRaw[c] = np.log10(dataRaw[c].astype(float))

    dataRaw.plot(figsize=(10, 5), fontsize=15, lw=4)
    plt.xlabel('Window', fontsize=15)
    
    if logScale:
        plt.ylabel('Log_10 of ' + metric, fontsize=15)
    else:
        plt.ylabel(metric, fontsize=15)
    plt.legend(loc=3)
    
    if logScale:
        plt.yscale('log')
    
    plt.savefig(experiment + '_' + exploration + '_' + metric + '.png')

# a function to concatenate performance metrics 
# from two subsequent blocks of epochs
# filepath_part_1 - filepath of the first epoch block
# filepath_part_2 - filepath of the second epoch block
# returns the two blocks combined
def concatEpochs(filepath_part_1, filepath_part_2):

    raw_data_part_1 = pd.read_csv(filepath_part_1)
    raw_data_part_2 = pd.read_csv(filepath_part_2)

    data_part_1_T = raw_data_part_1.T
    data_part_1_T.columns = data_part_1_T.iloc[0]
    data_part_1_T = data_part_1_T[1:]

    data_part_2_T = raw_data_part_2.T
    data_part_2_T.columns = data_part_2_T.iloc[0]
    data_part_2_T = data_part_2_T[1:]

    data_combined = pd.concat([data_part_1_T, data_part_2_T])

    return data_combined

# a function to plot the final total number of cells
# as boxplots over sliding windows of 20 epochs
# filepathRaw - filepath of the raw performance metrics
# experiment - indication of the experiment
# exploration - indication of the hyperparameter exploration
# combined - indicates whether twp epoch blocks must be combined (defaults to False)
# filepath2 - file path of the raw performance metrics of the second block of epochs (needed if combined=True)
def plotPerformance_raw_boxplot(filepathRaw, experiment, exploration, combined=False, filepath2=''):

    # combine data from the two epoch blocks if required
    if combined:
        dataRaw=concatEpochs(filepathRaw, filepath2)
    else:
        raw_dataRaw = pd.read_csv(filepathRaw)
        dataRaw_T = raw_dataRaw.T
        dataRaw_T.columns = dataRaw_T.iloc[0]
        dataRaw = dataRaw_T[1:]

    # slice data in sliding windows of n epochs, jumping over m*10 epochs
    # n: window size
    # m: multiplier to set the slice index right for selected n
    m = 1
    n = 21 * m
    # slice position data in window-based chunks
    movingAvg_df = [dataRaw[i:i + n] for i in range(0, len(dataRaw), 10 * m)]

    # generating one subplot per process, and using windows as columns.
    # generate new dataframe list
    # df: training process
    # rows: epochs
    # columns: windows
    exp=1
    # enforcing correspondance btween exp name and learning parameters in paper
    
    #EXP 1
    for training_name in ['lr=0.001 gamma=0.99', 'lr=0.001 gamma=0.95', 'lr=0.0001 gamma=0.99', 'lr=0.0001 gamma=0.95', 'lr=0.00001 gamma=0.99', 'lr=0.00001 gamma=0.95']:

        i=0

        training_df = pd.DataFrame()

        #print(training_name)

        for df in movingAvg_df:

            name=str(i)
            #print('series to be added\n', df[training_name])

            new_col = df[training_name].reset_index(drop=True)

            # each append is a new chunk corresponding to a new window
            training_df.insert(i, name, new_col)
            i=i+1
            #print(training_df)

        training_df = pd.DataFrame(training_df, dtype='float')
        
        fig, ax = plt.subplots()
        boxplot = training_df.boxplot(fontsize=12, showmeans = True, meanline = True, figsize=(30,20))
        plt.title('Exp'+str(exp)+ ' - ' + training_name, fontsize=20)
        plt.ylabel('Final number of cells', fontsize=15)
        plt.xlabel('Windows', fontsize=15)
        plt.yticks(np.arange(250, 271, 5))
        plt.savefig('Exp'+str(exp)+'_boxplot_'+training_name+'_'+experiment+'_'+exploration+'.png')
        plt.close()
        
        exp=exp+1


# a function to plot the generated protocols at the start and end epochs
# filepath_axis - filepath of the axis stimuli administered
# filepath_comprForce - filepath of the comprForce stimuli administered
# experiment - indication of the experiment
# exploration - indication of the hyperparameter exploration 
# start_epoch - starting epoch to consider to extract protocol
# stop_epoch - last epoch to consider to extract protocol
def plotProtocols(filepath_axis, filepath_comprForce, experiment, exploration, start_epoch, stop_epoch):

    data_raw_axis = pd.read_csv(filepath_axis)
    data_axis_T = data_raw_axis.T

    data_comprForce = pd.read_csv(filepath_comprForce)
    data_comprForce_T = data_comprForce.T

    data_axis_T.columns = data_axis_T.iloc[0]
    data_comprForce_T.columns = data_comprForce.iloc[0]

    start_epoch_name = str('epoch ' + str(start_epoch))
    stop_epoch_name = str('epoch ' + str(stop_epoch))

    protocol_start_epoch = pd.DataFrame(np.concatenate([data_axis_T[[start_epoch_name]], data_comprForce[[start_epoch_name]]], axis=1), columns=['Axis', 'ComprForce'])
    protocol_start_epoch = protocol_start_epoch[1:]
    protocol_stop_epoch = pd.DataFrame(np.concatenate([data_axis_T[[stop_epoch_name]], data_comprForce_T[[stop_epoch_name]]], axis=1), columns=['Axis', 'ComprForce'])
    protocol_stop_epoch = protocol_stop_epoch[1:]

    colors = {'X': 'tab:orange', 'Y': 'tab:blue'}

    fig, ax = plt.subplots()
    protocol_start_epoch['indexes'] = protocol_start_epoch.index
    protocol_start_epoch.plot(x='indexes', y='ComprForce', kind='scatter', figsize=(10, 5), fontsize=15, lw=4, c=protocol_start_epoch['Axis'].map(colors))
    plt.ylim([-0.001, 0.03])
    plt.xlabel('Simulation steps', fontsize=15)
    plt.ylabel('ComprForce', fontsize=15)
    plt.legend(loc='upper right')
    plt.savefig(experiment + '_' + exploration + '_' + start_epoch_name + '_protocol.png')

    protocol_stop_epoch['indexes'] = protocol_stop_epoch.index
    protocol_stop_epoch.plot(x='indexes', y='ComprForce', kind='scatter', figsize=(10, 5), fontsize=15, lw=4, c=protocol_stop_epoch['Axis'].map(colors))
    plt.ylim([-0.001, 0.03])
    plt.ylim([-0.001, 0.03])
    plt.xlabel('Simulation steps', fontsize=15)
    plt.ylabel('ComprForce', fontsize=15)
    plt.legend(loc='upper right')
    plt.savefig(experiment + '_' + exploration + '_' + stop_epoch_name + '_protocol.png')
    plt.close()

# a function to plot series of heatmaps showing explored initial positions in windows
# highlight corresponding performance
# start from disjoint subsequent epoch series
# fillTarget decides whether to fill the target with color intensity proportional to performance (mean) of considered window
def plotCoordinates(filepath_positions_part_1, filepath_metric_part_1, experiment, exploration, bestExperimentName, fillTarget=True, combined=False,filepath_positions_part_2="", filepath_metric_part_2=""):

    if combined:

        #metrics
        data_metrics=concatEpochs(filepath_metric_part_1, filepath_metric_part_2)

        #positions
        data_positions=concatEpochs(filepath_positions_part_1, filepath_positions_part_2)
    else:

        #metrics
        raw_metrics_data = pd.read_csv(filepath_metric_part_1)
        raw_metrics_data_T = raw_metrics_data.T
        raw_metrics_data_T.columns = raw_metrics_data_T.iloc[0]
        data_metrics = raw_metrics_data_T[1:]

        #positions
        raw_positions_data = pd.read_csv(filepath_positions_part_1)
        data_positions_T = raw_positions_data.T
        data_positions_T.columns = data_positions_T.iloc[0]
        data_positions = data_positions_T[1:]

    # selecting performance of best experiment
    best_experiment_performance = data_metrics[bestExperimentName]
    best_experiment_performance.columns=[bestExperimentName]

    # HEATMAPS
    # slice data in sliding windows of n epochs, jumping over m*10 epochs
    # n: window size
    # m: multiplier to set the slice index right for selected n
    m = 1
    n = 21 * m
    # slice position data in window-based chunks
    list_df = [data_positions[i:i + n] for i in range(0, len(data_positions), 10 * m)]
    #print(list_df)

    # set range of relevant windows
    # considering windows 1 - 12
    # starting from window 1
    # selecting df chunks from the second to two chunks before end since the last two are partial
    # using window-1 to compute colors on performance list cmap
    # TODO: enforce window range / performance list coherence
    window_range = [1, 12]
    window = window_range[0]

    for chunk in list_df[window_range[0]:window_range[1]+1]:

        #epoch indexes considered for computation
        print(chunk.index)

        if fillTarget:

            # create colormap with performance values
            performance_list = list(best_experiment_performance)
            min_val, max_val = min(performance_list), max(performance_list)

            # use the reds colormap that is built-in and normalize over performance values
            norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
            color = mpl.cm.Reds(norm(performance_list))

            circle = plt.Circle((200, 250), 80, color=color[window-1], fill=True)
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
        else:
            circle = plt.Circle((200, 250), 80, color='r', fill=False)
            fig, ax = plt.subplots()
            ax.set_aspect('equal')

        fig = plt.hist2d(x=chunk['x axis'], y=chunk['y axis'], bins=[8, 8], cmap='Purples', range=[[0, 400], [0, 400]])
        plt.title('Window ' + str(window), fontsize=30)
        plt.xlabel('X axis', fontsize=15)
        plt.ylabel('Y axis', fontsize=15)
        ax.tick_params(labelsize=12)
        ax.set_xbound(0, 400)
        ax.set_ybound(0, 400)
        # adding target
        ax.add_patch(circle)

        plt.savefig(str(window) + '_' + experiment + '_' + exploration + '_' +  bestExperimentName + '_heatmap_coordinates_combined.png')  # '_'+str(m)+
        plt.close()

        window = window + 1


if __name__ == '__main__':


    # TARGET 2
    results_folder="../..results/experiment2/"

    # Performance over epochs - final fraction of cells within target area (combining 2 sequential training moments)
    plotPerformance_final_cells(results_folder+"", experiment='2_final_frac_cells', exploration='lr_gamma', combined=True, filepath2=results_folder+"")

    # Performance over windows - boxplots per window
    plotPerformance_raw_boxplot(results_folder+"", experiment='2_final_frac_cells', exploration='lr_gamma', combined=True, filepath2=results_folder+"")

    # Performance over windows - MEAN of the final fraction of cells within target area (combining 2 sequential training moments)
    plotPerformance_metrics_lines(results_folder+"", experiment='2_final_frac_cells', exploration='lr_gamma', metric='Mean', combined=True, filepath2=results_folder+"")

    # Performance over windows - VARIANCE of the final fraction of cells within target area (combining 2 sequential training moments)
    plotPerformance_metrics_lines(results_folder+"", experiment='2_final_frac_cells', exploration='lr_gamma', metric='Variance', combined=True, filepath2=results_folder+"")

    # Protocol over epochs - compression stimuli protocols at epoch 0 and epoch N
    plotProtocols(results_folder+"", results_folder+"", start_epoch=0, stop_epoch=140)

    # Protocol over windows - explored positions and corresponding mean performance (combining 2 sequential training moments)
    plotCoordinates(filepath_metric_part_1=results_folder+"", filepath_metric_part_2=results_folder+"", filepath_positions_part_1=results_folder+"", filepath_positions_part_2=results_folder+"", experiment='2_final_fraction_cells', exploration='lr_gamma', bestExperimentName='lr=0.001 gamma=0.99')


    