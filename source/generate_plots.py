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
import math
import random
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.cm as cm


# a function to plot the final total number of cells
# results_folder, data_file : the filepath - filepath of the output .npy file
# experiment - indication of the experiment
# exploration - indication of the hyperparameter exploration
def plotPerformanceRL(results_folder, data_file, experiment, exploration, parameterValues, explore=True):   
    if experiment == "1_final_n_cells":

        data=pd.DataFrame.from_dict(np.load(results_folder+data_file, allow_pickle=True).item())
        cells = data['cell_numbers']

    elif experiment == "2_final_fraction_cells":

        data=pd.DataFrame.from_dict(np.load(results_folder+data_file, allow_pickle=True).item())
        #print(data.circle_actions[0])
        cells = data['inside_outside']
    
        # EXTRACT FRACTION OF CELLS INSIDE TARGET
        insideFraction=pd.DataFrame(columns=["inside"], index=range(len(cells)))
        for i, w in enumerate(cells):
            insideFraction.iloc[i]=w[0]
        cells=insideFraction["inside"]
        print("CELLS: ", cells)
    
    # PLOT RAW CELLS 
    cellPlot=sns.lineplot(data=cells, color= "purple")
    if experiment == '1_final_n_cells':
        plt.ylabel('Final number of cells', fontsize=10)
        cellPlot.set_title("Final number of cells - "+exploration+":"+parameterValues, fontsize=12)
    if experiment == '2_final_fraction_cells':
        plt.ylabel('Final fraction of cells inside target', fontsize=10)
        cellPlot.set_title("Final fraction of cells inside target - "+exploration+":"+parameterValues, fontsize=12)
    plt.xlabel('Epochs', fontsize=10)
    
    
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '.png')
    plt.close()

    # CREATE WINDOWS OUT OF EPOCHS
    # slice data in sliding windows of n epochs, jumping over m*10 epochs
    # n: window size
    # m: multiplier to set the slice index right for selected n
    m = 1
    n = 21 * m
    # slice position data in window-based chunks
    cellsWindows = [cells[i:i + n] for i in range(0, len(cells), 10 * m)]

    # COMPUTE METRICS
    metricValues=pd.DataFrame(index=range(len(cellsWindows)), columns=["Cells", "Mean", "Variance"])
    
    for i, w in enumerate(cellsWindows):
        content=list(w)
        mean=sum(content)/len(content)
        var=sum((i - mean)**2 for i in content)/len(content)

        metricValues["Cells"].iloc[i] = content
        metricValues["Mean"].iloc[i] = mean
        metricValues["Variance"].iloc[i] = var

    print(metricValues)
    if experiment == '1_final_n_cells' and len(metricValues)>5:
        metricValues.drop(index=[6,7], inplace=True)
    if experiment == '2_final_fraction_cells':
        metricValues.drop(index=[13,14], inplace=True)

    mean = metricValues['Mean']
    var = metricValues['Variance']

    print(metricValues)

    # PLOT MEAN
    meanPlot=sns.lineplot(data=mean, color= "purple")
    
    if experiment == '1_final_n_cells':
        plt.ylabel('Mean of the final number of cells', fontsize=10)
        meanPlot.set_title("Mean of the final numbers of cells - "+exploration+":"+parameterValues, fontsize=12)
    if experiment == '2_final_fraction_cells':
        plt.ylabel('Mean of the fraction of cells inside target', fontsize=10)
        meanPlot.set_title("Mean of the final fraction of cells inside target - "+exploration+":"+parameterValues, fontsize=12)

    plt.xlabel('Windows', fontsize=10)
    
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_MEAN.png')
    plt.close()

    # PLOT VARIANCE
    varPlot=sns.lineplot(data=var, color= "purple")

    if experiment == '1_final_n_cells':
        plt.ylabel('Variance of the final number of cells', fontsize=10)
        varPlot.set_title("Variance of final numbers of cells - "+exploration+":"+parameterValues, fontsize=12)
    if experiment == '2_final_fraction_cells':
        plt.ylabel('Variance of the final fraction of cells inside target', fontsize=10)
        varPlot.set_title("Variance of final fraction of cells inside target - "+exploration+":"+parameterValues, fontsize=12)
    plt.legend(loc=3)

    plt.xlabel('Windows', fontsize=10)
    
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_VARIANCE.png')
    plt.close()

    # EXTRACT CELLS IN WINDOWS
    cellsOverWindows=pd.DataFrame()
    for w in range(len(metricValues)):
        #print(metricValues["Cells"][w])
        cellsOverWindows[w]=pd.Series(metricValues["Cells"][w])
        #print(cellsOverWindows)

    # PLOT MAX FITNESS PER WINDOW
    maxFitness=cellsOverWindows.max()
    maxPlot=sns.scatterplot(data=maxFitness, color= "purple")

    if experiment == '1_final_n_cells':
        plt.ylabel('Variance of the final number of cells', fontsize=10)
        maxPlot.set_title("Maximum final number of cells - "+exploration+":"+parameterValues, fontsize=12)
    if experiment == '2_final_fraction_cells':
        plt.ylabel('Variance of the final fraction of cells inside target', fontsize=10)
        maxPlot.set_title("Maximum final fraction of cells inside target - "+exploration+":"+parameterValues, fontsize=12)

    plt.xlabel('Windows', fontsize=10)
    
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_MAX.png')
    plt.close()

    #TODO: make cols drop systematic when they are shorted than expected window
    #for c in cellsOverWindows.columns:
    #    if len(list(cellsOverWindows.loc[c].dropna()))<int(n/m):
    #        cellsOverWindows.drop(columns=[c], axis=1)
    #print(cellsOverWindows)
    #if experiment == '1_final_n_cells':
    #    cellsOverWindows.drop(columns=[6,7], inplace=True)
    #if experiment == '2_final_fraction_cells':
   #     cellsOverWindows.drop(columns=[13,14], inplace=True)
    

    # PLOT VIOLIN PLOTS PER WINDOW
    plot = sns.violinplot(data=cellsOverWindows, fill=True, split=False, color="yellow", inner="box", inner_kws=dict(box_width=4, whis_width=1), cut=0)
    
    plot.set_xlabel("Windows",  fontsize=10)
    if experiment == '1_final_n_cells':
        plot.set_ylabel("Final number of cells", fontsize=10)
        plot.set_title("Final number of cells - "+exploration+":"+parameterValues, fontsize=12)
    if experiment == '2_final_fraction_cells':
        plot.set_ylabel("Final fraction of cells inside target", fontsize=10)
        plot.set_title("Final fraction of cells inside target - "+exploration+":"+parameterValues, fontsize=12)

    if exploration=="lr_gamma":
        plt.ylim(252, 271)

    if exploration=="numIter":
        plt.ylim(264, 279)

    
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_violin_boxplots.png')
    plt.close()



# a function to plot the final total number of cells
# results_folder, data_file : the filepath - filepath of the output .npy file
# experiment - indication of the experiment
# exploration - indication of the hyperparameter exploration
def plotPerformanceGA(results_folder, data_file, experiment, exploration, parameterValues):
    
    fitnesses=pd.read_csv(results_folder+data_file)
    cells=fitnesses["cells"]

    # DF WITH GENERATIONS AS COLS
    cellsOverGenerations=pd.DataFrame()
    for w in range(len(fitnesses)):
        #print(cells.iloc[w])
        cellsOverGenerations[w]= pd.Series(cells.iloc[w].split(";")).astype(int)
    

    # PLOT MAX FITNESS PER GENERATION
    maxFitness=cellsOverGenerations.max()
    maxFPlot=sns.scatterplot(data=maxFitness)
    maxFPlot.set_title("MAX FITNESS OVER GENERATIONS " + experiment + '_' + exploration + "_" + parameterValues)
    maxFPlot.set_xlabel("Generations", fontsize=10)
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_MAX.png')
    plt.close()

    
    metricValues=pd.DataFrame(index=range(len(cellsOverGenerations.columns)), columns=["Cells", "Mean", "Variance"])
    
    for i in range(len(cellsOverGenerations.columns)):
        cells=pd.Series(cellsOverGenerations[i])
        metricValues["Cells"].iloc[i] = list(cells)
        metricValues["Mean"].iloc[i] = cells.mean()
        metricValues["Variance"].iloc[i] = cells.var()
    #print(metricValues)

    # PLOT MEAN OF FITNESS PER GENERATION
    means=metricValues["Mean"]
    vars=metricValues["Variance"]
    meanPlot=sns.lineplot(data=means)
    if experiment == '1_final_n_cells':
        plt.ylabel('Mean of the final number of cells', fontsize=10)
        meanPlot.set_title("Maximum final number of cells - "+exploration+":"+parameterValues, fontsize=12)
    if experiment == '2_final_fraction_cells':
        plt.ylabel('Mean of the final fraction of cells inside target', fontsize=10)
        meanPlot.set_title("Maximum final fraction of cells inside target - "+exploration+":"+parameterValues, fontsize=12)
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_MEAN.png')
    plt.close()

    # PLOT VARIANCE OF FITNESS PER GENERATION
    varPlot=sns.lineplot(data=vars)
    if experiment == '1_final_n_cells':
        plt.ylabel('Variance of the final number of cells', fontsize=10)
        varPlot.set_title("Maximum final number of cells - "+exploration+":"+parameterValues, fontsize=12)
    if experiment == '2_final_fraction_cells':
        plt.ylabel('Variance of the final fraction of cells inside target', fontsize=10)
        varPlot.set_title("Maximum final fraction of cells inside target - "+exploration+":"+parameterValues, fontsize=12)
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_VARIANCE.png')
    plt.close()

    plot = sns.violinplot(data=cellsOverGenerations, fill=True, split=False, color="yellow", inner="box", inner_kws=dict(box_width=4, whis_width=1), cut=0)
    
    plot.set_xlabel("Generations",  fontsize=10)
    if experiment == '1_final_n_cells':
        plot.set_ylabel("Final number of cells", fontsize=10)
        plot.set_title("Final number of cells - "+exploration+":"+parameterValues, fontsize=12)
    if experiment == '2_final_fraction_cells':
        plot.set_ylabel("Final fraction of cells inside target", fontsize=10)
        plot.set_title("Final fraction of cells inside target - "+exploration+":"+parameterValues, fontsize=12)

    #if exploration=="numIter":
    #    plt.ylim(264, 279)
    
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_violin_boxplots.png')
    plt.close()


def plotTestingResults(results_folder, data_file, experiment, exploration, parameterValues, logScale=False):

    data=pd.DataFrame.from_dict(np.load(results_folder+"/testing/"+data_file, allow_pickle=True).item())

    # SELECT TESTING EPOCHS
    cells = data['cell_numbers'][70:]

    # PLOT HISTOGRAM OF FITNESS VALUES ACROSS TESTING EPOCHS
    
    testPlot = sns.histplot(data=cells, fill=False, binwidth=1, color="purple")
    testPlot.set_title("Distribution of final number of cells - "+exploration+":"+parameterValues, fontsize=12)
    testPlot.set_xlabel("Final number of cells",  fontsize=10)
    testPlot.set_xlabel("Count",  fontsize=10)
    plt.savefig(results_folder +"/"+ experiment + '_' + exploration + "_" + parameterValues + '_TEST.png')
    plt.close()

# a function to plot the generated protocols at the start and end epochs
# results_folder, data_file : the filepath - filepath of the output .npy file including information on axis and comprForce stimuli administered
# experiment - indication of the experiment
# exploration - indication of the hyperparameter exploration 
# start_epoch - starting epoch to consider to extract protocol
# stop_epoch - last epoch to consider to extract protocol
def plotProtocols(results_folder, data_file, experiment, exploration, start_epoch, stop_epoch):

    data=pd.DataFrame.from_dict(np.load(results_folder+data_file, allow_pickle=True).item())

    # TAKE PROTOCOLS FROM TWO EPOCHS
    start_epoch_name = str('epoch_' + str(start_epoch))
    stop_epoch_name = str('epoch_' + str(stop_epoch))
    protocol1=pd.DataFrame(data["compr_history"][start_epoch], columns=["axis", "comprForce"])
    protocol2=pd.DataFrame(data["compr_history"][stop_epoch], columns=["axis", "comprForce"])

    colors = {'X': 'tab:orange', 'Y': 'tab:blue'}

    # PLOT PROTOCOLS
    fig, ax = plt.subplots()
    protocol1['indexes'] = protocol1.index
    protocol1.plot(x='indexes', y='comprForce', kind='scatter', figsize=(10, 5), fontsize=15, lw=4, c=protocol1['axis'].map(colors))
    #plt.ylim([-0.001, 0.01])
    plt.xlabel('Learning episodes', fontsize=10)
    plt.ylabel('comprForce', fontsize=10)
    plt.legend(loc='upper right')
    plt.savefig(results_folder+experiment + '_' + exploration + "_" + start_epoch_name + '_protocol.png')
    plt.close()

    protocol2['indexes'] = protocol2.index
    protocol2.plot(x='indexes', y='comprForce', kind='scatter', figsize=(10, 5), fontsize=15, lw=4, c=protocol2['axis'].map(colors))
    #plt.ylim([-0.001, 0.01])
    plt.xlabel('Learning episodes', fontsize=10)
    plt.ylabel('comprForce', fontsize=10)
    plt.legend(loc='upper right')
    plt.savefig(results_folder+experiment + '_' + exploration + "_" + stop_epoch_name + '_protocol.png')
    plt.close()

def plotInitialPositions(results_folder, data_file, experiment, exploration, bestExperimentName, fillTarget=True):

    data=pd.DataFrame.from_dict(np.load(results_folder+data_file, allow_pickle=True).item())
    positions = data['xxx']

    best_experiment_performance = positions[bestExperimentName]
    best_experiment_performance.positions=[bestExperimentName]
    
    # HEATMAPS
    # slice data in sliding windows of n epochs, jumping over m*10 epochs
    # n: window size
    # m: multiplier to set the slice index right for selected n
    m = 1
    n = 21 * m
    # slice position data in window-based chunks
    list_df = [positions[i:i + n] for i in range(0, len(positions), 10 * m)]
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
        #print(chunk.index)

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
        plt.xlabel('X axis',  fontsize=10)
        plt.ylabel('Y axis',  fontsize=10)
        ax.tick_params(labelsize=10)
        ax.set_xbound(0, 400)
        ax.set_ybound(0, 400)
        # adding target
        ax.add_patch(circle)

        plt.savefig(str(window) + '_' + experiment + '_' + exploration + '_' +  bestExperimentName + '_heatmap_coordinates_combined.png')
        plt.close()

        window = window + 1



if __name__ == '__main__':

    # EXPERIMENT 1.1 - RL - TARGET 1 - LR GAMMA
    results_folder="results/experiment1.1/"
    epoch="70"
    numIter="20"
    for lr in ["0.001", "0.0001", "1e-05"]:
        for gamma in ["0.95", "0.99"]:

            plotPerformanceRL(results_folder=results_folder+"new_palacell_out_"+lr+"_"+gamma+"/", data_file="data_to_save_at_epoch_"+epoch+".npy", experiment='1_final_n_cells', exploration='lr_gamma', parameterValues=lr+"_"+gamma)
    
    # EXPERIMENT 1.2 - RL - TARGET 1 - NUMITER
    results_folder="results/experiment1.2/"
    gamma="0.99"
    lr="0.001"
    for numIter in ["20", "40", "50", "100", "200"]:
        
        epoch="70"
        plotPerformanceRL(results_folder=results_folder+"new_palacell_out_iters_"+numIter+"_"+lr+"_"+gamma+"/", data_file="data_to_save_at_epoch_"+epoch+".npy", experiment='1_final_n_cells', exploration='numIter', parameterValues=numIter)

        #epoch="99(best)"
        #plotTestingResults(results_folder=results_folder+"new_palacell_out_iters_"+numIter+"_"+lr+"_"+gamma, data_file="data_to_save_at_epoch_"+epoch+".npy", experiment='1_final_n_cells', exploration='numIter', parameterValues=numIter)

    # EXPERIMENT 2 - RL - TARGET 2 - LR GAMMA
    results_folder="results/experiment2/"
    epoch="140"
    numIter="20"
    experiment='2_final_fraction_cells'

    for lr in ["0.001", "0.0001", "1e-05"]:
        for gamma in ["0.95", "0.99"]:
            pass
            #plotPerformanceRL(results_folder=results_folder+"new_palacell_out_circles_"+lr+"_"+gamma+"/", data_file="data_to_save_at_epoch_"+epoch+".npy", experiment='2_final_fraction_cells', exploration='lr_gamma', parameterValues=lr+"_"+gamma)

    # EXPERIMENT 3 - GA - TARGET 1 - NUMITER
    results_folder="results/experiment3/"
    for numIter in ["20", "50", "100", "200"]:
        
        plotPerformanceGA(results_folder=results_folder+"new_palacell_out_"+numIter+".0/", data_file="fitness_track.csv", experiment='1_final_n_cells', exploration='numIter', parameterValues=numIter)
    
