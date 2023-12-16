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
def plotPerformanceRLCellNumber(results_folder, data_file, experiment, exploration, parameterValues)

    data=pd.DataFrame.from_dict(np.load(results_folder+data_file, allow_pickle=True).item())
    cells = data['cell_numbers']
    
    # PLOT RAW CELLS 
    cellPlot=sns.lineplot(data=cells)
    if experiment == '1_final_n_cells':
        plt.ylabel('Final number of cells', fontsize=15)
    if experiment == '2_final_fraction_cells':
        plt.ylabel('Final fraction of cells inside target', fontsize=15)
    cellPlot.set_title("FITNESS OVER EPOCHS" + experiment + '_' + exploration + "_" + parameterValues)
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
        metricValues["Cells"].iloc[i] = list(w)
        metricValues["Mean"].iloc[i] = w.mean()
        metricValues["Variance"].iloc[i] = w.var()

    mean = metricValues['Mean']
    var = metricValues['Variance']

    # PLOT MEAN
    meanPlot=sns.lineplot(data=mean)
    meanPlot.set_title("MEAN FITNESS OVER WINDOWS " + experiment + '_' + exploration + "_" + parameterValues)  
    if experiment == '1_final_n_cells':
        plt.ylabel('Mean of the final number of cells', fontsize=15)
    if experiment == '2_final_fraction_cells':
        plt.ylabel('Mean of the fraction of cells inside target', fontsize=15) 
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_MEAN.png')
    plt.close()

    # PLOT VARIANCE
    varPlot=sns.lineplot(data=var)
    varPlot.set_title("VARIANCE OF FITNESS OVER WINDOWS " + experiment + '_' + exploration + "_" + parameterValues) 
    if experiment == '1_final_n_cells':
        plt.ylabel('Variance of the final number of cells', fontsize=15)
    if experiment == '2_final_fraction_cells':
        plt.ylabel('Variance of the final fraction of cells inside target', fontsize=15)
    plt.legend(loc=3)
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_VARIANCE.png')
    plt.close()

    # EXTRACT CELLS IN WINDOWS
    cellsOverWindows=pd.DataFrame()
    for w in range(len(metricValues)):
        print(metricValues["Cells"][w])
        cellsOverWindows[w]=pd.Series(metricValues["Cells"][w])
        print(cellsOverWindows)

    # PLOT MAX FITNESS PER WINDOW
    maxFitness=cellsOverWindows.max()
    maxPlot=sns.scatterplot(data=maxFitness)
    maxPlot.set_title("MAX FITNESS OVER WINDOWS " + experiment + '_' + exploration + "_" + parameterValues)
    maxPlot.set_xlabel("Generations")
    #plot.set(ylim=(260, 285))
    #sns.despine(left=True, bottom=True)
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_MAX.png')
    plt.close()

    #TODO: make cols drop systematic when they are shorted than expected window
    #for c in cellsOverWindows.columns:
    #    if len(list(cellsOverWindows.loc[c].dropna()))<int(n/m):
    #        cellsOverWindows.drop(columns=[c], axis=1)

    cellsOverWindows.drop(columns=[6,7], inplace=True)
    print(cellsOverWindows)

    # PLOT VIOLIN PLOTS PER WINDOW
    plot = sns.violinplot(data=cellsOverWindows, bw_adjust=.5, cut=1, linewidth=1, fill=False, linecolor='b')
    plot.set_title("FITNESS DISTRIBUTIONS OVER WINDOWS " + experiment + '_' + exploration + "_" + parameterValues)
    #plot.set_ylim(245, 280)
    plot.set_xlabel("Windows")
    if experiment == '1_final_n_cells':
        plot.set_ylabel("Final number of cells")
    if experiment == '2_final_fraction_cells':
        plot.set_ylabel("Final fraction of cells inside target")

    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_violin_boxplots.png')
    plt.close()

# a function to plot the final fraction of cells inside target
# results_folder, data_file : the filepath - filepath of the output .npy file
# experiment - indication of the experiment
# exploration - indication of the hyperparameter exploration
def plotPerformanceRLCircles(results_folder, data_file, experiment, exploration, parameterValues):

    data=pd.DataFrame.from_dict(np.load(results_folder+data_file, allow_pickle=True).item())
    print(data.circle_actions[0])
    cells = data['inside_outside']
    
    # EXTRACT FRACTION OF CELLS INSIDE TARGET
    insideFraction=pd.DataFrame(columns=["inside"], index=range(len(cells)))
    for i, w in enumerate(cells):
        insideFraction.iloc[i]=w[0]

    # PLOT FRACTION OF CELLS INSIDE
    cellPlot=sns.lineplot(data=insideFraction)
    if experiment == '1_final_n_cells':
        plt.ylabel('Final number of cells', fontsize=15)
    if experiment == '2_final_fraction_cells':
        plt.ylabel('Final fraction of cells inside target', fontsize=15)
    cellPlot.set_title("FITNESS OVER EPOCHS" + experiment + '_' + exploration + "_" + parameterValues)
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '.png')
    plt.close()

    # CREATE WINDOWS OUT OF EPOCHS
    # slice data in sliding windows of n epochs, jumping over m*10 epochs
    # n: window size
    # m: multiplier to set the slice index right for selected n
    m = 1
    n = 21 * m
    # slice position data in window-based chunks
    cellsWindows = [insideFraction[i:i + n] for i in range(0, len(insideFraction), 10 * m)]

    # COMPUTE METRICS
    metricValues=pd.DataFrame(index=range(len(cellsWindows)), columns=["Cells", "Mean", "Variance"])

    print(type(cellsWindows[0]))
    
    for i, w in enumerate(cellsWindows):
        print(w, type(w))
        content=list(w['inside'])
        print(content)
        mean=sum(content)/len(content)
        var=sum((i - mean)**2 for i in content) / len(content)
        metricValues["Cells"].iloc[i] = content
        metricValues["Mean"].iloc[i] = mean
        metricValues["Variance"].iloc[i] = var
    
    mean = metricValues['Mean']
    var = metricValues['Variance']

    # PLOT MEAN OF THE FRACTION OF CELLS INSIDE
    meanPlot=sns.lineplot(data=mean)
    meanPlot.set_title("MEAN FITNESS OVER WINDOWS " + experiment + '_' + exploration + "_" + parameterValues)  
    if experiment == '1_final_n_cells':
        plt.ylabel('Mean of the final number of cells', fontsize=15)
    if experiment == '2_final_fraction_cells':
        plt.ylabel('Mean of the fraction of cells inside target', fontsize=15) 
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_MEAN.png')
    plt.close()

    # PLOT VARIANCE OF THE FRACTION OF CELLS INSIDE
    varPlot=sns.lineplot(data=var)
    varPlot.set_title("VARIANCE OF FITNESS OVER WINDOWS " + experiment + '_' + exploration + "_" + parameterValues) 
    if experiment == '1_final_n_cells':
        plt.ylabel('Variance of the final number of cells', fontsize=15)
    if experiment == '2_final_fraction_cells':
        plt.ylabel('Variance of the final fraction of cells inside target', fontsize=15)
    plt.legend(loc=3)
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_VARIANCE.png')
    plt.close()

    # EXTRACT THE FRACTION OF CELLS INSIDE PER WINDOW
    cellsOverWindows=pd.DataFrame()
    for w in range(len(metricValues)):
        print(metricValues["Cells"][w])
        cellsOverWindows[w]=pd.Series(metricValues["Cells"][w])
        print(cellsOverWindows)

    # PLOT MAX OF THE FRACTION OF CELLS INSIDE PER WINDOW
    maxFitness=cellsOverWindows.max()
    maxPlot=sns.scatterplot(data=maxFitness)
    maxPlot.set_title("MAX FITNESS OVER WINDOWS " + experiment + '_' + exploration + "_" + parameterValues)
    maxPlot.set_xlabel("Generations")
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_MAX.png')
    plt.close()

    #TODO: make cols drop systematic when they are shorted than expected window
    #for c in cellsOverWindows.columns:
    #    if len(list(cellsOverWindows.loc[c].dropna()))<int(n/m):
    #        cellsOverWindows.drop(columns=[c], axis=1)

    #cellsOverWindows.drop(columns=[6,7], inplace=True)

    # PLOT VIOLIN PLOTS OF THE FRACTION OF CELLS INSIDE
    violin = sns.violinplot(data=cellsOverWindows)#, bw_adjust=.5, cut=1, linewidth=1, palette="Set3")
    plot=violin
    plot.set_title("FITNESS DISTRIBUTIONS OVER WINDOWS " + experiment + '_' + exploration + "_" + parameterValues)
    plot.set_xlabel("Windows")
    if experiment == '1_final_n_cells':
        plot.set_ylabel("Final number of cells")
    if experiment == '2_final_fraction_cells':
        plot.set_ylabel("Final fraction of cells inside target")
    
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_violin_boxplots.png')
    plt.close()

# a function to plot the final total number of cells
# results_folder, data_file : the filepath - filepath of the output .npy file
# experiment - indication of the experiment
# exploration - indication of the hyperparameter exploration
def plotPerformanceGACellNumber(results_folder, data_file, experiment, exploration, parameterValues):
    fitnesses=pd.read_csv(results_folder+data_file)
    cells=fitnesses["cells"]

    # DF WITH GENERATIONS AS COLS
    cellsOverGenerations=pd.DataFrame()
    for w in range(len(fitnesses)):
        print(cells.iloc[w])
        cellsOverGenerations[w]= pd.Series(cells.iloc[w].split(";")).astype(int)
    
    # PLOT VIOLINS PER GENERATION
    cellPlot = sns.violinplot(data=cellsOverGenerations)#, palette="Set3")
    cellPlot.set_title("FITNESS DISTRIBUTIONS OVER GENERATIONS " + experiment + '_' + exploration + "_" + parameterValues)
    cellPlot.set_xlabel("Generations")
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_violin_boxplots.png')
    plt.close()

    # PLOT MAX FITNESS PER GENERATION
    maxFitness=cellsOverGenerations.max()
    maxFPlot=sns.scatterplot(data=maxFitness)
    maxFPlot.set_title("MAX FITNESS OVER GENERATIONS " + experiment + '_' + exploration + "_" + parameterValues)
    maxFPlot.set_xlabel("Generations")
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_MAX.png')
    plt.close()

    
    metricValues=pd.DataFrame(index=range(len(cellsOverGenerations.columns)), columns=["Cells", "Mean", "Variance"])
    
    for i in range(len(cellsOverGenerations.columns)):
        cells=pd.Series(cellsOverGenerations[i])
        metricValues["Cells"].iloc[i] = list(cells)
        metricValues["Mean"].iloc[i] = cells.mean()
        metricValues["Variance"].iloc[i] = cells.var()
    print(metricValues)

    # PLOT MEAN OF FITNESS PER GENERATION
    means=metricValues["Mean"]
    vars=metricValues["Variance"]
    meanPlot=sns.lineplot(data=means)
    meanPlot.set_title("MEAN FITNESS OVER WINDOWS " + experiment + '_' + exploration + "_" + parameterValues)
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_MEAN.png')
    plt.close()

    # PLOT VARIANCE OF FITNESS PER GENERATION
    varPlot=sns.lineplot(data=vars)
    varPlot.set_title("VARIANCE OF FITNESS OVER WINDOWS " + experiment + '_' + exploration + "_" + parameterValues)
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_VARIANCE.png')
    plt.close()


def plotTestingResults(results_folder, data_file, experiment, exploration, parameterValues, logScale=False):

    data=pd.DataFrame.from_dict(np.load(results_folder+"/testing/"+data_file, allow_pickle=True).item())

    # SELECT TESTING EPOCHS
    cells = data['cell_numbers'][70:]

    # PLOT HISTOGRAM OF FITNESS VALUES ACROSS TESTING EPOCHS
    testPlot = sns.histplot(data=cells, binwidth=1)
    testPlot.set_title("MAX FITNESS OVER TESTING EPOCH " + experiment + '_' + exploration + "_" + parameterValues)
    testPlot.set_xlabel("Final number of cells")
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_TEST.png')
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
    plt.xlabel('Learning episodes', fontsize=15)
    plt.ylabel('comprForce', fontsize=15)
    plt.legend(loc='upper right')
    plt.savefig(results_folder+experiment + '_' + exploration + "_" + start_epoch_name + '_protocol.png')
    plt.close()

    protocol2['indexes'] = protocol2.index
    protocol2.plot(x='indexes', y='comprForce', kind='scatter', figsize=(10, 5), fontsize=15, lw=4, c=protocol2['axis'].map(colors))
    #plt.ylim([-0.001, 0.01])
    plt.xlabel('Learning episodes', fontsize=15)
    plt.ylabel('comprForce', fontsize=15)
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
    print(list_df)

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

        plt.savefig(str(window) + '_' + experiment + '_' + exploration + '_' +  bestExperimentName + '_heatmap_coordinates_combined.png')
        plt.close()

        window = window + 1



if __name__ == '__main__':

    results_folder="results/experiment2/"
    lr=str(1e-05)
    gamma=str(0.95)
    epoch="20"
    experiment='2_final_fraction_cells'

    plotPerformanceRLCircles(results_folder=results_folder+"new_palacell_out_circles_"+lr+"_"+gamma+"/", data_file="data_to_save_at_epoch_"+epoch+".npy", experiment='1_final_n_cells', exploration='lr_gamma', parameterValues=lr+"_"+gamma)
    
    results_folder="results/experiment1.2/"
    lr="0.0001"
    gamma="0.99"
    numIter="100"
    epoch="70"
    # Performance over epochs - final number of cells
    plotPerformanceRLCellNumber(results_folder=results_folder+"new_palacell_out_iters_"+numIter+"_"+lr+"_"+gamma+"/", data_file="data_to_save_at_epoch_"+epoch+".npy", experiment='1_final_n_cells', exploration='numIter', parameterValues=numIter)
    
    exit(0)
    #TARGET 1 - GA RESULTS 
    results_folder="results/experiment3/"
    numIter="50.0"


    #analyzeProtocolsGA(results_folder=results_folder+"new_palacell_out_"+numIter+"/", data_file="output.csv", experiment='3_GA_1.1', exploration='numIter',parameterValues=numIter)    
    results_folder="results/experiment1.1/"
    lr=str(0.0001)
    gamma=str(0.99)
    
    analyzeProtocols(results_folder=results_folder+"new_palacell_out_"+lr+"_"+gamma+"/", data_file="data_to_save_at_epoch_70.npy", experiment='exp1.1', exploration='lr_gamma', parameterValues=lr+"_"+gamma)
    exit(0)
    # Performance over windows - MEAN and VARIANCE of the final number of cells lineplot
    plotPerformanceGA(results_folder=results_folder+"new_palacell_out_"+numIter+"/", data_file="fitness_track.csv", experiment='3_GA_1.1', exploration='numIter',parameterValues=numIter)

    
    # TARGET 1 - lr and gamma exploration
    results_folder="results/experiment1.1/"
    epoch="70"
    for lr in ["1e-05","0.001", "0.0001", "1e-05"]:
        for gamma in ["0.95", "0.99"]:

            # Performance over epochs - final number of cells
            plotPerformanceCellNumber(results_folder=results_folder+"new_palacell_out_"+lr+"_"+gamma+"/", data_file="data_to_save_at_epoch_"+epoch+".npy", experiment='1_final_n_cells', exploration='lr_gamma', parameterValues=lr+"_"+gamma)

            # Performance over windows - MEAN and VARIANCE of the final number of cells lineplot
            plotPerformanceMetrics(results_folder=results_folder+"new_palacell_out_"+lr+"_"+gamma+"/", data_file="data_to_save_at_epoch_70.npy", experiment='1_final_n_cells', exploration='lr_gamma', logScale=False, parameterValues=lr+"_"+gamma)       

            # Protocol over epochs - compression stimuli protocols at epoch 0 and epoch N
            for i in range(0, 70, 70):
                plotProtocols(results_folder=results_folder+"new_palacell_out_"+lr+"_"+gamma+"/", data_file="data_to_save_at_epoch_70.npy", experiment='exp1.1', exploration='lr_gamma', start_epoch=0, stop_epoch=i)
            

    # TARGET 1 - numIters exploration
    results_folder="results/experiment1.2/"
    lr="0.0001"
    gamma="0.99"

    for numIter in ["20", "50", "100", "200"]:

        epoch="70"
        # Performance over epochs - final number of cells
        plotPerformanceCellNumber(results_folder=results_folder+"new_palacell_out_iters_"+numIter+"_"+lr+"_"+gamma+"/", data_file="data_to_save_at_epoch_"+epoch+".npy", experiment='1_final_n_cells', exploration='numIter', parameterValues=numIter)

        # Performance over windows - MEAN and VARIANCE of the final number of cells lineplot
        plotPerformanceMetrics(results_folder=results_folder+"new_palacell_out_iters_"+numIter+"_"+lr+"_"+gamma+"/", data_file="data_to_save_at_epoch_70.npy", experiment='1_final_n_cells', exploration='numIter', logScale=False, parameterValues=numIter)       

        #plot testing results
        epoch="99(best)"
        plotTestingResults(results_folder=results_folder+"new_palacell_out_iters_"+numIter+"_"+lr+"_"+gamma+"/", data_file="data_to_save_at_epoch_"+epoch+".npy", experiment='1_final_n_cells', exploration='numIter', parameterValues=numIter)

        # Protocol over epochs - compression stimuli protocols at epoch 0 and epoch N
        for i in range(0, 70, 70):
            plotProtocols(results_folder=results_folder+"new_palacell_out_iters_"+numIter+"_"+lr+"_"+gamma+"/", data_file="data_to_save_at_epoch_70.npy", experiment='exp1.1', exploration='numIter', start_epoch=0, stop_epoch=i)
            
