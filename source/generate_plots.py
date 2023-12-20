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
def plotPerformanceRL(results_folder, data_file, experiment, exploration, parameterValues):   
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
        plt.ylabel('Final number of cells')
        #cellPlot.set_title("Final number of cells - "+exploration+":"+parameterValues)
    if experiment == '2_final_fraction_cells':
        plt.ylabel('Final fraction of cells inside target')
        #cellPlot.set_title("Final fraction of cells inside target - "+exploration+":"+parameterValues)
    plt.xlabel('Epochs')
    
    
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

    # selecting only the first 6 windows
    cellsWindows=cellsWindows[:6]

    # COMPUTE METRICS
    metricValues=pd.DataFrame(index=range(len(cellsWindows)), columns=["Cells", "Mean", "Variance"])
    
    for i, w in enumerate(cellsWindows):
        content=list(w)
        mean=sum(content)/len(content)
        var=sum((i - mean)**2 for i in content)/len(content)

        metricValues["Cells"].iloc[i] = content
        metricValues["Mean"].iloc[i] = mean
        metricValues["Variance"].iloc[i] = var

    

    mean = metricValues['Mean']
    var = metricValues['Variance']

    # PLOT MEAN

    #fig = plt.figure(figsize=(12, 8))
    meanPlot=sns.lineplot(data=mean, color= "darkorange", markers=True)
    
    if experiment == '1_final_n_cells':
        plt.ylabel('Mean of the final number of cells')
        #meanPlot.set_title("Mean of the final numbers of cells - "+exploration+":"+parameterValues)
    if experiment == '2_final_fraction_cells':
        plt.ylabel('Mean of the fraction of cells inside target')
        #meanPlot.set_title("Mean of the final fraction of cells inside target - "+exploration+":"+parameterValues)
        

    plt.xlabel('Windows')
    
    
    if experiment == '2_final_fraction_cells' and exploration=="numIter":
        plt.ylim(0.58, 0.63)
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_MEAN.png')
    plt.close()

    # PLOT VARIANCE
    varPlot=sns.lineplot(data=var, color= "purple")

    if experiment == '1_final_n_cells':
        plt.ylabel('Variance of the final number of cells')
        #varPlot.set_title("Variance of final numbers of cells - "+exploration+":"+parameterValues)
    if experiment == '2_final_fraction_cells':
        plt.ylabel('Variance of the final fraction of cells inside target')
        #varPlot.set_title("Variance of final fraction of cells inside target - "+exploration+":"+parameterValues)
    plt.legend(loc=3)

    plt.xlabel('Windows')
    
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_VARIANCE.png')
    plt.close()

    # EXTRACT CELLS IN WINDOWS
    cellsOverWindows=pd.DataFrame()
    for w in range(len(metricValues)):
        cellsOverWindows[w]=pd.Series(metricValues["Cells"][w]).astype(float)

    # PLOT MAX FITNESS PER WINDOW
    maxFitness=cellsOverWindows.max()
    maxPlot=sns.scatterplot(data=maxFitness, color= "purple")

    if experiment == '1_final_n_cells':
        plt.ylabel('Variance of the final number of cells')
        #maxPlot.set_title("Maximum final number of cells - "+exploration+":"+parameterValues)
    if experiment == '2_final_fraction_cells':
        plt.ylabel('Variance of the final fraction of cells inside target')
        #maxPlot.set_title("Maximum final fraction of cells inside target - "+exploration+":"+parameterValues)

    plt.xlabel('Windows')
    
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_MAX.png')
    plt.close()

    # PLOT VIOLIN PLOTS PER WINDOW
    #fig = plt.figure(figsize=(12, 6))
    vplot = sns.violinplot(data=cellsOverWindows, fill=True, split=False, color="darkorange", linecolor="black", width=.9)
    plot = sns.boxplot(data=cellsOverWindows, width=0.2, notch=False, fill=True, color= "seashell", linecolor="black", ax=vplot)
    
    plot.set_xlabel("Windows")
    if experiment == '1_final_n_cells':
        plot.set_ylabel("Final number of cells")
        #plot.set_title("Final number of cells - "+exploration+":"+parameterValues)
    if experiment == '2_final_fraction_cells':
        plot.set_ylabel("Final fraction of cells inside target")
        #plot.set_title("Final fraction of cells inside target - "+exploration+":"+parameterValues)

    if experiment == '1_final_n_cells' and exploration=="lr_gamma":
        plt.ylim(247, 275) 
        
    if experiment == '1_final_n_cells' and exploration=="numIter":
        plt.ylim(247, 284)

    if experiment == '2_final_fraction_cells' and exploration=="lr_gamma":
        plt.ylim(0.55, 0.67) #R 80
    
    if experiment == '2_final_fraction_cells' and exploration=="numIter":
        plt.ylim(0.56, 0.66)
        
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_violin_boxplots.png')
    plt.close()



# a function to plot the final total number of cells
# results_folder, data_file : the filepath - filepath of the output .npy file
# experiment - indication of the experiment
# exploration - indication of the hyperparameter exploration
def plotPerformanceGA(results_folder, data_file, experiment, exploration, parameterValues):
    
    fitnesses=pd.read_csv(results_folder+data_file)
    cells=fitnesses["Cells"]

    # DF WITH GENERATIONS AS COLS
    cellsOverGenerations=pd.DataFrame()
    for w in range(len(fitnesses)):
        #print(cells.iloc[w])
        cellsOverGenerations[w]= pd.Series(cells.iloc[w].split(";")).astype(float)

    # selecting only the first 9 generations
    cellsOverGenerations=cellsOverGenerations.iloc[:, :9]
    

    # PLOT MAX FITNESS PER GENERATION
    maxFitness=cellsOverGenerations.max()
    maxFPlot=sns.scatterplot(data=maxFitness, color= "magenta")
    #maxFPlot.set_title("MAX FITNESS OVER GENERATIONS " + experiment + '_' + exploration + "_" + parameterValues)
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

    #fig = plt.figure(figsize=(12, 6))

    
    meanPlot=sns.lineplot(data=means, color= "deepskyblue")

    if experiment == '1_final_n_cells':
        plt.ylabel('Mean of the final number of cells')
        #meanPlot.set_title("Maximum final number of cells - "+exploration+":"+parameterValues)
    if experiment == '2_final_fraction_cells':
        plt.ylabel('Mean of the final fraction of cells inside target')
        #meanPlot.set_title("Maximum final fraction of cells inside target - "+exploration+":"+parameterValues)
        
        
    if experiment == '2_final_fraction_cells' and exploration=="numIter":
        plt.ylim(0.58, 0.63)

    plt.xlabel('Generations')
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_MEAN.png')
    plt.close()

    # PLOT VARIANCE OF FITNESS PER GENERATION
    varPlot=sns.lineplot(data=vars, color= "magenta")
    if experiment == '1_final_n_cells':
        plt.ylabel('Variance of the final number of cells')
        #varPlot.set_title("Maximum final number of cells - "+exploration+":"+parameterValues)
    if experiment == '2_final_fraction_cells':
        plt.ylabel('Variance of the final fraction of cells inside target')
        #varPlot.set_title("Maximum final fraction of cells inside target - "+exploration+":"+parameterValues)
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_VARIANCE.png')
    plt.close()

    
    #fig = plt.figure(figsize=(12, 6))
    vplot = sns.violinplot(data=cellsOverGenerations, fill=True, split=False, color="deepskyblue", linecolor="black", width=.8)
    plot = sns.boxplot(data=cellsOverGenerations, width=0.2, notch=False, fill=True, color= "azure", linecolor="black", ax=vplot)
    
    plot.set_xlabel("Generations")
    if experiment == '1_final_n_cells':
        plot.set_ylabel("Final number of cells")
        #plot.set_title("Final number of cells - "+exploration+":"+parameterValues)
    if experiment == '2_final_fraction_cells':
        plot.set_ylabel("Final fraction of cells inside target")
        #plot.set_title("Final fraction of cells inside target - "+exploration+":"+parameterValues)

    if experiment == '2_final_fraction_cells' and exploration=="numIter":
        plt.ylim(0.56, 0.66)

    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_violin_boxplots.png')
    plt.close()


def plotTestingResults(results_folder, data_file, experiment, exploration, parameterValues, logScale=False):

    
    if experiment == "1_final_n_cells":

        data=pd.DataFrame.from_dict(np.load(results_folder+"/"+data_file, allow_pickle=True).item())
        cells = data['cell_numbers']
        # SELECT TESTING EPOCHS
        cells = cells[70:]
        print(numIter, cells.mean(), cells.max())
        cellList = list(cells)
        step = 1
        start = np.floor(min(cellList) / step) * step
        stop = max(cellList) + step
        bin_edges = np.arange(254, 279, step=step)
        plt.hist(cellList, bins=bin_edges, color='crimson', ec='white')
        plt.xticks(list(bin_edges), rotation=45)
        plt.xlim([254, 279])
        plt.ylim([0,30])
        #plt.title("Final fraction of cells in testing - "+exploration+":"+parameterValues)
        plt.ylabel("Count",  fontsize=10)
        plt.savefig(results_folder +"/"+ experiment + '_' + exploration + "_" + parameterValues + '_TEST.png')
        plt.close()

    elif experiment == "2_final_fraction_cells":

        data=pd.DataFrame.from_dict(np.load(results_folder+data_file, allow_pickle=True).item())
        cells = data['inside_outside']
    
        # EXTRACT FRACTION OF CELLS INSIDE TARGET
        insideFraction=pd.DataFrame(columns=["inside"], index=range(len(cells)))
        for i, w in enumerate(cells):
            insideFraction.iloc[i]=w[0]
        cells=insideFraction["inside"]
        # SELECT TESTING EPOCHS
        cells = cells[70:]
        print(numIter, cells.mean(), cells.max())

        cellList = list(cells)
        step = 0.005
        start = np.floor(min(cellList) / step) * step
        stop = max(cellList) + step
        bin_edges = np.arange(start, stop, step=step)
        plt.hist(cellList, bins=bin_edges, color='crimson', ec='white')
        plt.xticks(list(bin_edges), rotation=45)
        plt.xlim([0.58, 0.64])
        plt.ylim([0,10])
        #plt.title("Final fraction of cells in testing - "+exploration+":"+parameterValues)
        plt.ylabel("Count",  fontsize=10)
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

    protocol1["comprForce"]=protocol1["comprForce"].astype(float)
    protocol2["comprForce"]=protocol2["comprForce"].astype(float)


    # PLOT PROTOCOLS
    fig, ax = plt.subplots()
    protocol1['indexes'] = protocol1.index
    protocol1.plot(x='indexes', y='comprForce', kind='scatter', lw=4, c=protocol1['axis'].map(colors))
    plt.ylim([-0.001, 0.025])
    plt.xlabel('Learning episodes', fontsize=15)
    plt.ylabel('comprForce', fontsize=15)
    plt.legend(loc='upper right')
    plt.savefig(results_folder+experiment + '_' + exploration + "_" + start_epoch_name + '_protocol.png')
    plt.close()

    protocol2['indexes'] = protocol2.index
    protocol2.plot(x='indexes', y='comprForce', kind='scatter', lw=4, c=protocol2['axis'].map(colors))
    plt.ylim([-0.001, 0.025])
    plt.xlabel('Learning episodes', fontsize=15)
    plt.ylabel('comprForce', fontsize=15)
    plt.legend(loc='upper right')
    plt.savefig(results_folder+experiment + '_' + exploration + "_" + stop_epoch_name + '_protocol.png')
    plt.close()

def plotInitialPositions(results_folder, data_file, experiment, exploration, fillTarget=True):

    data=pd.DataFrame.from_dict(np.load(results_folder+data_file, allow_pickle=True).item())
    
    #create df linking positions and performances
    pos_df=pd.DataFrame(index=range(len(data)), columns=["x-coord", "y-coord", "fitness"])
    
    positions = data['circle_actions']
    for i, position in enumerate(positions):
        print(i, position[0].numpy(),position[1].numpy(), data["inside_outside"][i][0])
        pos_df["x-coord"].iloc[i]= position[0].numpy()
        pos_df["y-coord"].iloc[i]= position[1].numpy()
        pos_df["fitness"].iloc[i]= data["inside_outside"].iloc[i][0]
    
    # HEATMAPS
    # slice data in sliding windows of n epochs, jumping over m*10 epochs
    # n: window size
    # m: multiplier to set the slice index right for selected n
    m = 1
    n = 21 * m
    # slice position data in window-based chunks
    list_df = [pos_df[i:i + n] for i in range(0, len(pos_df), 10 * m)]
    print(list_df)

    # set range of relevant windows
    # considering windows 1 - 6
    # starting from window 1
    # selecting df chunks from the second to two chunks before end since the last two are partial
    # using window-1 to compute colors on performance list cmap
    # TODO: enforce window range / performance list coherence
    window_range = [0, 6]
    window = window_range[0]

    performances=[]
    for df in list_df:
        performances.append(df["fitness"].mean())

    # create colormap with performance values
    min_val, max_val = min(performances), max(performances)
    # use the reds colormap that is built-in and normalize over performance values
    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    color = mpl.cm.Reds(norm(performances))
    
    for chunk in list_df[window_range[0]:window_range[1]]:

        #epoch indexes considered for computation
        print(chunk.index)

        if fillTarget:

            circle = plt.Circle((200, 250), 80, color=color[window], fill=True)
            fig, ax = plt.subplots()
            ax.set_aspect('equal')

        else:

            circle = plt.Circle((200, 250), 80, color='r', fill=False)
            fig, ax = plt.subplots()
            ax.set_aspect('equal')

        fig = plt.hist2d(x=chunk["x-coord"], y=chunk["y-coord"], bins=[8, 8], cmap='Purples', range=[[0, 400], [0, 400]])
        #plt.title('Window ' + str(window), fontsize=30)
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        #ax.tick_params()
        ax.set_xbound(0, 400)
        ax.set_ybound(0, 400)
        # adding target
        ax.add_patch(circle)

        plt.savefig(results_folder + experiment + '_' + exploration + "_" + str(window) + '_heatmap_coordinates.png')
        plt.close()

        window = window + 1



if __name__ == '__main__':

    #setting general plotting style
    sns.set_context(font_scale=1.5)
    sns.set_style("whitegrid")

    # EXPERIMENT 1.2 - RL - TARGET 1 - NUMITER
    results_folder="results/experiment1.2_final/"
    gamma="0.99"
    lr="0.0001"
    for numIter in ["20", "50", "100", "200"]:
        
        epoch="70"
        plotPerformanceRL(results_folder=results_folder+"new_palacell_out_iters_"+numIter+"_"+lr+"_"+gamma+"/", data_file="data_to_save_at_epoch_"+epoch+".npy", experiment='1_final_n_cells', exploration='numIter', parameterValues=numIter)

        epoch="99(best)"
        plotTestingResults(results_folder=results_folder+"new_palacell_out_iters_"+numIter+"_"+lr+"_"+gamma+"/testing/", data_file="data_to_save_at_epoch_"+epoch+".npy", experiment='1_final_n_cells', exploration='numIter', parameterValues=numIter)

            

    #exit(0)

    # EXPERIMENT 2 - RL - TARGET 2 - ITERS
    results_folder="results/experiment2_iters/"
    epoch="70"
    gamma="0.95"
    lr="0.0001"
    experiment='2_final_fraction_cells'

    for numIter in ["20", "50", "100"]:
            
            plotInitialPositions(results_folder=results_folder+"new_palacell_out_circles_iters_"+numIter+"_"+lr+"_"+gamma+"/", data_file="data_to_save_at_epoch_"+epoch+".npy", experiment='2_final_fraction_cells', exploration='numIter')

    # EXPERIMENT 1.1 - RL - TARGET 1 - LR GAMMA
    results_folder="results/experiment1.1_final/"
    epoch="70"
    numIter="20"
    for lr in ["0.001", "0.0001", "1e-05"]:
        for gamma in ["0.95", "0.99"]:

            plotPerformanceRL(results_folder=results_folder+"new_palacell_out_"+lr+"_"+gamma+"/", data_file="data_to_save_at_epoch_"+epoch+".npy", experiment='1_final_n_cells', exploration='lr_gamma', parameterValues=lr+"_"+gamma)

            plotProtocols(results_folder=results_folder+"new_palacell_out_"+lr+"_"+gamma+"/", data_file="data_to_save_at_epoch_"+epoch+".npy", experiment='1_final_n_cells', exploration='lr_gamma', start_epoch=0, stop_epoch=37)
    
    # EXPERIMENT 1.2 - RL - TARGET 1 - NUMITER
    results_folder="results/experiment1.2_final/"
    gamma="0.99"
    lr="0.0001"
    for numIter in ["20", "50", "100", "200"]:
        
        epoch="70"
        plotPerformanceRL(results_folder=results_folder+"new_palacell_out_iters_"+numIter+"_"+lr+"_"+gamma+"/", data_file="data_to_save_at_epoch_"+epoch+".npy", experiment='1_final_n_cells', exploration='numIter', parameterValues=numIter)

        #epoch="99(best)"
        #plotTestingResults(results_folder=results_folder+"new_palacell_out_iters_"+numIter+"_"+lr+"_"+gamma, data_file="data_to_save_at_epoch_"+epoch+".npy", experiment='1_final_n_cells', exploration='numIter', parameterValues=numIter)

    # EXPERIMENT 2 - RL - TARGET 2 - LR GAMMA
    results_folder="results/experiment2_radius80_final/"
    epoch="70"
    numIter="20"
    experiment='2_final_fraction_cells'

    for lr in ["0.001", "0.0001", "1e-05"]:
        for gamma in ["0.95", "0.99"]:
            
            plotPerformanceRL(results_folder=results_folder+"new_palacell_out_circles_"+lr+"_"+gamma+"/", data_file="data_to_save_at_epoch_"+epoch+".npy", experiment='2_final_fraction_cells', exploration='lr_gamma', parameterValues=lr+"_"+gamma)

    # EXPERIMENT 2 - RL - TARGET 2 - ITERS
    results_folder="results/experiment2_iters/"
    epoch="70"
    gamma="0.95"
    lr="0.0001"
    experiment='2_final_fraction_cells'

    for numIter in ["20", "50", "100"]:
            
            plotPerformanceRL(results_folder=results_folder+"new_palacell_out_circles_iters_"+numIter+"_"+lr+"_"+gamma+"/", data_file="data_to_save_at_epoch_"+epoch+".npy", experiment='2_final_fraction_cells', exploration='numIter', parameterValues=numIter+"_"+lr+"_"+gamma)

            epoch="99(best)"
            plotTestingResults(results_folder=results_folder+"new_palacell_out_circles_iters_"+numIter+"_"+lr+"_"+gamma+"/testing/", data_file="data_to_save_at_epoch_"+epoch+".npy", experiment='2_final_fraction_cells', exploration='numIter', parameterValues=numIter+"_"+lr+"_"+gamma)
            
            epoch="70"
            plotInitialPositions(results_folder=results_folder+"new_palacell_out_circles_iters_"+numIter+"_"+lr+"_"+gamma+"/", data_file="data_to_save_at_epoch_"+epoch+".npy", experiment='2_final_fraction_cells', exploration='numIter')



    # EXPERIMENT 4 - GA - TARGET 2 - NUMITER
    results_folder="results/experiment4_final/"
    for numIter in ["20", "50", "100", "200"]:
        
        plotPerformanceGA(results_folder=results_folder+"new_palacell_out_"+numIter+".0/", data_file="fitness_track.csv", experiment='2_final_fraction_cells', exploration='numIter', parameterValues=numIter)

    # EXPERIMENT 4_R25 - GA - TARGET 2 - NUMITER
    results_folder="results/experiment4_r25/"
    for numIter in ["20", "50", "100", "200"]:
        
        plotPerformanceGA(results_folder=results_folder+"new_palacell_out_"+numIter+".0/", data_file="fitness_track.csv", experiment='2_final_fraction_cells', exploration='numIter', parameterValues=numIter)