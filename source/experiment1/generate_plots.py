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
# results_folder, data_file : the filepath - filepath of the output .npy file
# experiment - indication of the experiment
# exploration - indication of the hyperparameter exploration
def plotPerformanceCellNumber(results_folder, data_file, experiment, exploration, parameterValues):

    if exploration != 'lr_gamma' and exploration != 'numIter':
        print('Insert valid exploration name! Either lr_gamma or numIter')

    data=pd.DataFrame.from_dict(np.load(results_folder+data_file, allow_pickle=True).item())

    cells = data['cell_numbers']

    cells.plot(figsize=(10, 5), fontsize=15, lw=4)
    plt.xlabel('Epoch', fontsize=15)
    if experiment == '1_final_n_cells':
        plt.ylabel('Final number of cells', fontsize=15)
    if experiment == '2_final_fraction_cells':
        plt.ylabel('Final fraction of cells inside target', fontsize=15)
    plt.legend(loc=3)
    plt.savefig(results_folder+experiment + '_' + exploration + "_" + parameterValues + '.png')
    plt.close()

# a function to plot performance metrics - mean and variance
# results_folder, data_file : the filepath - filepath of the output .npy file
# experiment - indication of the experiment
# exploration - indication of the hyperparameter exploration 
# metric - the metric to consider (mean or variance)
# logScale - provides logScale visualization on the y axis if True (defaults to False)
def plotPerformanceMetrics(results_folder, data_file, experiment, exploration, metric, parameterValues, logScale=False):
    
    if exploration != 'lr_gamma' and exploration != 'numIter':
        print('Insert valid exploration name! Either lr_gamma or numIter')

    data=pd.DataFrame.from_dict(np.load(results_folder+data_file, allow_pickle=True).item())

    cells = data['cell_numbers']

    # slice data in sliding windows of n epochs, jumping over m*10 epochs
    # n: window size
    # m: multiplier to set the slice index right for selected n
    m = 1
    n = 21 * m
    # slice position data in window-based chunks
    cellsWindows = [cells[i:i + n] for i in range(0, len(cells), 10 * m)]

    metricValues=pd.DataFrame(index=range(len(cellsWindows)), columns=["Cells", "Mean", "Variance"])
    
    for i, w in enumerate(cellsWindows):
        metricValues["Cells"].iloc[i] = list(w)
        metricValues["Mean"].iloc[i] = w.mean()
        metricValues["Variance"].iloc[i] = w.var()

    print(metricValues)
    
    mean = metricValues['Mean']
    var = metricValues['Variance']

    mean.plot(figsize=(10, 5), fontsize=15, lw=4)
    plt.xlabel('Epoch', fontsize=15)
    if experiment == '1_final_n_cells':
        plt.ylabel('Mean of the final number of cells', fontsize=15)
    if experiment == '2_final_fraction_cells':
        plt.ylabel('Mean of the fraction of cells inside target', fontsize=15)
    plt.legend(loc=3)
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_MEAN.png')
    plt.close()

    var.plot(figsize=(10, 5), fontsize=15, lw=4)
    plt.xlabel('Epoch', fontsize=15)
    if experiment == '1_final_n_cells':
        plt.ylabel('Variance of the final number of cells', fontsize=15)
    if experiment == '2_final_fraction_cells':
        plt.ylabel('Variance of the final fraction of cells inside target', fontsize=15)
    plt.legend(loc=3)
    plt.savefig(results_folder+experiment + '_' + exploration + "_" + parameterValues + '_VARIANCE.png')
    plt.close()

    #if logScale:
        #TODO adapt logscale to specific columns


# a function to plot the generated protocols at the start and end epochs
# results_folder, data_file : the filepath - filepath of the output .npy file including information on axis and comprForce stimuli administered
# experiment - indication of the experiment
# exploration - indication of the hyperparameter exploration 
# start_epoch - starting epoch to consider to extract protocol
# stop_epoch - last epoch to consider to extract protocol
def plotProtocols(results_folder, data_file, experiment, exploration, start_epoch, stop_epoch):

    if exploration != 'lr_gamma' and exploration != 'numIter':
        print('Insert valid exploration name! Either lr_gamma or numIter')

    data=pd.DataFrame.from_dict(np.load(results_folder+data_file, allow_pickle=True).item())

    start_epoch_name = str('epoch_' + str(start_epoch))
    stop_epoch_name = str('epoch_' + str(stop_epoch))

    protocol1=pd.DataFrame(data["compr_history"][start_epoch], columns=["axis", "comprForce"])
    protocol2=pd.DataFrame(data["compr_history"][stop_epoch], columns=["axis", "comprForce"])

    print(protocol1, protocol2)



    colors = {'X': 'tab:orange', 'Y': 'tab:blue'}

    fig, ax = plt.subplots()
    protocol1['indexes'] = protocol1.index
    protocol1.plot(x='indexes', y='comprForce', kind='scatter', figsize=(10, 5), fontsize=15, lw=4, c=protocol1['axis'].map(colors))
    #plt.ylim([-0.001, 0.01])
    plt.xlabel('Learning episodes', fontsize=15)
    plt.ylabel('comprForce', fontsize=15)
    plt.legend(loc='upper right')
    plt.savefig(results_folder+experiment + '_' + exploration + "_" + start_epoch_name + '_protocol.png')

    protocol2['indexes'] = protocol2.index
    protocol2.plot(x='indexes', y='comprForce', kind='scatter', figsize=(10, 5), fontsize=15, lw=4, c=protocol2['axis'].map(colors))
    #plt.ylim([-0.001, 0.01])
    plt.xlabel('Learning episodes', fontsize=15)
    plt.ylabel('comprForce', fontsize=15)
    plt.legend(loc='upper right')
    plt.savefig(results_folder+experiment + '_' + exploration + "_" + stop_epoch_name + '_protocol.png')
    plt.close()



if __name__ == '__main__':

    results_folder="results/experiment1.1/"
    # TARGET 1 - lr and gamma exploration

    epoch="70"

    for lr in ["0.001", "0.0001", "1e-05"]:
        for gamma in ["0.99", "0.95"]:


            # Performance over epochs - final number of cells
            plotPerformanceCellNumber(results_folder=results_folder+"new_palacell_out_"+lr+"_"+gamma+"/", data_file="data_to_save_at_epoch_"+epoch+".npy", experiment='1_final_n_cells', exploration='lr_gamma', parameterValues=lr+"_"+gamma)

            

            # Performance over windows - final number of cells boxplots
            #plotPerformance_raw_boxplot(results_folder+"new_palacell_out_"+lr+"_"+gamma+"/data_to_save_at_epoch_70.npy", experiment='1_final_n_cells', exploration='lr_gamma')


            # Performance over windows - MEAN of the final number of cells lineplot
            plotPerformanceMetrics(results_folder=results_folder+"new_palacell_out_"+lr+"_"+gamma+"/", data_file="data_to_save_at_epoch_70.npy", experiment='1_final_n_cells', exploration='lr_gamma', metric='Mean', logScale=False, parameterValues=lr+"_"+gamma)
            

            # Performance over windows - VARIANCE of the final number of cells
            #plotPerformance_metrics_lines(results_folder+"new_palacell_out_"+lr+"_"+gamma+"/data_to_save_at_epoch_70.npy", experiment='1_final_n_cells', exploration='lr_gamma', metric='Variance', logScale=True)

            # Protocol over epochs - compression stimuli protocols at epoch 0 and epoch N
            for i in range(70):
                plotProtocols(results_folder=results_folder+"new_palacell_out_"+lr+"_"+gamma+"/", data_file="data_to_save_at_epoch_70.npy", experiment='exp1.1', exploration='lr_gamma', start_epoch=0, stop_epoch=i)
            exit(0)

    # TARGET 1 - numIters exploration
    results_folder="results/experiment1.2/"

    for numiter in ["20", "50", "80", "100", "200"]:

        # Performance over epochs - final number of cells
        plotPerformance_final_cells(results_folder+"new_palacell_out_"+lr+"_"+gamma+"/data_to_save_at_epoch_70.npy", experiment='1_final_n_cells', exploration='numIter')

        # Performance over windows - final number of cells boxplots
        plotPerformance_raw_boxplot(results_folder+"new_palacell_out_"+lr+"_"+gamma+"/data_to_save_at_epoch_70.npy", experiment='1_final_n_cells', exploration='numIter')

        # Performance over windows - MEAN of the final number of cells
        plotPerformance_metrics_lines(results_folder+"new_palacell_out_"+lr+"_"+gamma+"/data_to_save_at_epoch_70.npy", experiment='1_final_n_cells', exploration='numIter', metric='Mean', logScale=False)

        # Performance over windows - VARIANCE of the final number of cells
        plotPerformance_metrics_lines(results_folder+"new_palacell_out_"+lr+"_"+gamma+"/data_to_save_at_epoch_70.npy", experiment='1_final_n_cells', exploration='numIter', metric='Variance', logScale=False)

    # Protocol over epochs - compression stimuli protocols at epoch 0 and epoch N
    #plotProtocols(results_folder+"path to protocol at start", results_folder+"path to protocol at stop", 'exp1.2')
