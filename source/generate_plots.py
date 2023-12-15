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
    cellPlot=sns.lineplot(data=cells)
    if experiment == '1_final_n_cells':
        plt.ylabel('Final number of cells', fontsize=15)
    if experiment == '2_final_fraction_cells':
        plt.ylabel('Final fraction of cells inside target', fontsize=15)
    cellPlot.set_title("FITNESS OVER EPOCHS" + experiment + '_' + exploration + "_" + parameterValues)
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '.png')
    plt.close()

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

    mean = metricValues['Mean']
    var = metricValues['Variance']

    meanPlot=sns.lineplot(data=mean)
    meanPlot.set_title("MEAN FITNESS OVER WINDOWS " + experiment + '_' + exploration + "_" + parameterValues)  
    if experiment == '1_final_n_cells':
        plt.ylabel('Mean of the final number of cells', fontsize=15)
    if experiment == '2_final_fraction_cells':
        plt.ylabel('Mean of the fraction of cells inside target', fontsize=15) 
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_MEAN.png')
    plt.close()

    varPlot=sns.lineplot(data=var)
    varPlot.set_title("VARIANCE OF FITNESS OVER WINDOWS " + experiment + '_' + exploration + "_" + parameterValues) 
    if experiment == '1_final_n_cells':
        plt.ylabel('Variance of the final number of cells', fontsize=15)
    if experiment == '2_final_fraction_cells':
        plt.ylabel('Variance of the final fraction of cells inside target', fontsize=15)
    plt.legend(loc=3)
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_VARIANCE.png')
    plt.close()

    cellsOverWindows=pd.DataFrame()
    for w in range(len(metricValues)):
        print(metricValues["Cells"][w])
        cellsOverWindows[w]=pd.Series(metricValues["Cells"][w])
        print(cellsOverWindows)

    #plotting max fitness per generation
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

    violin = sns.violinplot(data=cellsOverWindows)#, bw_adjust=.5, cut=1, linewidth=1, palette="Set3")
    plot=violin
    #violin = sns.violinplot(data=cellsOverWindows, palette='turbo', inner=None, linewidth=0, saturation=0.4)
    #boxplot = sns.boxplot(data=cellsOverWindows, palette='turbo', width=0.3,
    #boxprops={'zorder': 2}, ax=violin)
    plot.set_title("FITNESS DISTRIBUTIONS OVER WINDOWS " + experiment + '_' + exploration + "_" + parameterValues)
    #plot.set_ylim(245, 280)
    plot.set_xlabel("Windows")
    if experiment == '1_final_n_cells':
        plot.set_ylabel("Final number of cells")
    if experiment == '2_final_fraction_cells':
        plot.set_ylabel("Final fraction of cells inside target")

    #plot.set(ylim=(250, 275))
    #sns.despine(left=True, bottom=True)
    
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_violin_boxplots.png')
    plt.close()

# a function to plot the final fraction of cells inside target
# results_folder, data_file : the filepath - filepath of the output .npy file
# experiment - indication of the experiment
# exploration - indication of the hyperparameter exploration
def plotPerformanceCellsInside(results_folder, data_file, experiment, exploration, parameterValues):

    if exploration != 'lr_gamma' and exploration != 'numIter':
        print('Insert valid exploration name! Either lr_gamma or numIter')

    data=pd.DataFrame.from_dict(np.load(results_folder+data_file, allow_pickle=True).item())
    print(data.circle_actions[0])
    exit(0)
    cells = data['inside_outside']
    print(cells)
    
    insideFraction=pd.DataFrame(columns=["inside"], index=range(len(cells)))
    for i, w in enumerate(cells):
        insideFraction.iloc[i]=w[0]
    print(insideFraction)

    cellPlot=sns.lineplot(data=insideFraction)
    if experiment == '1_final_n_cells':
        plt.ylabel('Final number of cells', fontsize=15)
    if experiment == '2_final_fraction_cells':
        plt.ylabel('Final fraction of cells inside target', fontsize=15)
    cellPlot.set_title("FITNESS OVER EPOCHS" + experiment + '_' + exploration + "_" + parameterValues)
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '.png')
    plt.close()

    # slice data in sliding windows of n epochs, jumping over m*10 epochs
    # n: window size
    # m: multiplier to set the slice index right for selected n
    m = 1
    n = 21 * m
    # slice position data in window-based chunks
    cellsWindows = [insideFraction[i:i + n] for i in range(0, len(insideFraction), 10 * m)]

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
    
    print(metricValues)

    mean = metricValues['Mean']
    var = metricValues['Variance']

    meanPlot=sns.lineplot(data=mean)
    meanPlot.set_title("MEAN FITNESS OVER WINDOWS " + experiment + '_' + exploration + "_" + parameterValues)  
    if experiment == '1_final_n_cells':
        plt.ylabel('Mean of the final number of cells', fontsize=15)
    if experiment == '2_final_fraction_cells':
        plt.ylabel('Mean of the fraction of cells inside target', fontsize=15) 
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_MEAN.png')
    plt.close()

    varPlot=sns.lineplot(data=var)
    varPlot.set_title("VARIANCE OF FITNESS OVER WINDOWS " + experiment + '_' + exploration + "_" + parameterValues) 
    if experiment == '1_final_n_cells':
        plt.ylabel('Variance of the final number of cells', fontsize=15)
    if experiment == '2_final_fraction_cells':
        plt.ylabel('Variance of the final fraction of cells inside target', fontsize=15)
    plt.legend(loc=3)
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_VARIANCE.png')
    plt.close()

    cellsOverWindows=pd.DataFrame()
    for w in range(len(metricValues)):
        print(metricValues["Cells"][w])
        cellsOverWindows[w]=pd.Series(metricValues["Cells"][w])
        print(cellsOverWindows)

    #plotting max fitness per generation
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

    #cellsOverWindows.drop(columns=[6,7], inplace=True)
    print(cellsOverWindows)

    violin = sns.violinplot(data=cellsOverWindows)#, bw_adjust=.5, cut=1, linewidth=1, palette="Set3")
    plot=violin
    #violin = sns.violinplot(data=cellsOverWindows, palette='turbo', inner=None, linewidth=0, saturation=0.4)
    #boxplot = sns.boxplot(data=cellsOverWindows, palette='turbo', width=0.3,
    #boxprops={'zorder': 2}, ax=violin)
    plot.set_title("FITNESS DISTRIBUTIONS OVER WINDOWS " + experiment + '_' + exploration + "_" + parameterValues)
    #plot.set_ylim(245, 280)
    plot.set_xlabel("Windows")
    if experiment == '1_final_n_cells':
        plot.set_ylabel("Final number of cells")
    if experiment == '2_final_fraction_cells':
        plot.set_ylabel("Final fraction of cells inside target")

    #plot.set(ylim=(250, 275))
    #sns.despine(left=True, bottom=True)
    
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_violin_boxplots.png')
    plt.close()


# a function to plot performance metrics - mean and variance
# results_folder, data_file : the filepath - filepath of the output .npy file
# experiment - indication of the experiment
# exploration - indication of the hyperparameter exploration 
# metric - the metric to consider (mean or variance)
# logScale - provides logScale visualization on the y axis if True (defaults to False)
def plotPerformanceMetrics(results_folder, data_file, experiment, exploration, parameterValues, logScale=False):
    
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

    mean = metricValues['Mean']
    var = metricValues['Variance']

    meanPlot=sns.lineplot(data=mean)
    meanPlot.set_title("MEAN FITNESS OVER WINDOWS " + experiment + '_' + exploration + "_" + parameterValues)  
    if experiment == '1_final_n_cells':
        plt.ylabel('Mean of the final number of cells', fontsize=15)
    if experiment == '2_final_fraction_cells':
        plt.ylabel('Mean of the fraction of cells inside target', fontsize=15) 
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_MEAN.png')
    plt.close()

    varPlot=sns.lineplot(data=var)
    varPlot.set_title("VARIANCE OF FITNESS OVER WINDOWS " + experiment + '_' + exploration + "_" + parameterValues) 
    if experiment == '1_final_n_cells':
        plt.ylabel('Variance of the final number of cells', fontsize=15)
    if experiment == '2_final_fraction_cells':
        plt.ylabel('Variance of the final fraction of cells inside target', fontsize=15)
    plt.legend(loc=3)
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_VARIANCE.png')
    plt.close()

    cellsOverWindows=pd.DataFrame()
    for w in range(len(metricValues)):
        print(metricValues["Cells"][w])
        cellsOverWindows[w]=pd.Series(metricValues["Cells"][w])
        print(cellsOverWindows)

    #plotting max fitness per generation
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

    violin = sns.violinplot(data=cellsOverWindows)#, bw_adjust=.5, cut=1, linewidth=1, palette="Set3")
    plot=violin
    #violin = sns.violinplot(data=cellsOverWindows, palette='turbo', inner=None, linewidth=0, saturation=0.4)
    #boxplot = sns.boxplot(data=cellsOverWindows, palette='turbo', width=0.3,
    #boxprops={'zorder': 2}, ax=violin)
    plot.set_title("FITNESS DISTRIBUTIONS OVER WINDOWS " + experiment + '_' + exploration + "_" + parameterValues)
    #plot.set_ylim(245, 280)
    plot.set_xlabel("Windows")
    if experiment == '1_final_n_cells':
        plot.set_ylabel("Final number of cells")
    if experiment == '2_final_fraction_cells':
        plot.set_ylabel("Final fraction of cells inside target")

    #plot.set(ylim=(250, 275))
    #sns.despine(left=True, bottom=True)
    
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_violin_boxplots.png')
    plt.close()


# a function to plot the final total number of cells
# results_folder, data_file : the filepath - filepath of the output .npy file
# experiment - indication of the experiment
# exploration - indication of the hyperparameter exploration
def plotPerformanceGA(results_folder, data_file, experiment, exploration, parameterValues):

    if exploration != 'lr_gamma' and exploration != 'numIter':
        print('Insert valid exploration name! Either lr_gamma or numIter')

    print(results_folder+data_file)
    fitnesses=pd.read_csv(results_folder+data_file)

    cells=fitnesses["cells"]

    #obtaining df with generations as columns
    cellsOverGenerations=pd.DataFrame()
    for w in range(len(fitnesses)):
        print(cells.iloc[w])
        cellsOverGenerations[w]= pd.Series(cells.iloc[w].split(";")).astype(int)
    
    #plotting violinplot for each generation
    cellPlot = sns.violinplot(data=cellsOverGenerations)#, palette="Set3")
    cellPlot.set_title("FITNESS DISTRIBUTIONS OVER GENERATIONS " + experiment + '_' + exploration + "_" + parameterValues)
    cellPlot.set_xlabel("Generations")
    #plot.set(ylim=(260, 285))
    #sns.despine(left=True, bottom=True)
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_violin_boxplots.png')
    plt.close()

    #plotting max fitness per generation
    maxFitness=cellsOverGenerations.max()
    maxFPlot=sns.scatterplot(data=maxFitness)
    maxFPlot.set_title("MAX FITNESS OVER GENERATIONS " + experiment + '_' + exploration + "_" + parameterValues)
    maxFPlot.set_xlabel("Generations")
    #plot.set(ylim=(260, 285))
    #sns.despine(left=True, bottom=True)
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_MAX.png')
    plt.close()

    #plotting average and variance per generation
    metricValues=pd.DataFrame(index=range(len(cellsOverGenerations.columns)), columns=["Cells", "Mean", "Variance"])
    
    for i in range(len(cellsOverGenerations.columns)):
        cells=pd.Series(cellsOverGenerations[i])
        metricValues["Cells"].iloc[i] = list(cells)
        metricValues["Mean"].iloc[i] = cells.mean()
        metricValues["Variance"].iloc[i] = cells.var()
    print(metricValues)

    means=metricValues["Mean"]
    vars=metricValues["Variance"]
    meanPlot=sns.lineplot(data=means)
    meanPlot.set_title("MEAN FITNESS OVER WINDOWS " + experiment + '_' + exploration + "_" + parameterValues)
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_MEAN.png')
    plt.close()
    varPlot=sns.lineplot(data=vars)
    varPlot.set_title("VARIANCE OF FITNESS OVER WINDOWS " + experiment + '_' + exploration + "_" + parameterValues)
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_VARIANCE.png')
    plt.close()

def analyzeProtocolsGA(results_folder, data_file, experiment, exploration, parameterValues):

    output=pd.read_csv(results_folder+data_file, delimiter=",")
    protocols=output["Solution"]
    comprValues=pd.DataFrame()
    print(len(protocols))
    compr=[]
    for row in protocols:

        protocol = row.split(";")
        
        
        for stimuli in protocol:
            try:
                #print(stimuli.split("|")[1].strip("'").strip("]").strip("'"))
                compr.append(float(stimuli.split("|")[1].strip("'").strip("]").strip(" ").strip("'")))
            except Exception as e:
                print(e)


            
    print(len(compr))
    print("Average:", sum(compr)/len(compr))
    print("Max", max(compr))
    print("Min", min(compr))

    comprPlot=sns.histplot(data=compr)
    plt.show()



def plotTestingResults(results_folder, data_file, experiment, exploration, parameterValues, logScale=False):

    if exploration != 'lr_gamma' and exploration != 'numIter':
        print('Insert valid exploration name! Either lr_gamma or numIter')

    data=pd.DataFrame.from_dict(np.load(results_folder+"/testing/"+data_file, allow_pickle=True).item())
    cells = data['cell_numbers'][70:]

    testPlot = sns.histplot(data=cells, binwidth=1)
    testPlot.set_title("MAX FITNESS OVER TESTING EPOCH " + experiment + '_' + exploration + "_" + parameterValues)
    testPlot.set_xlabel("Final number of cells")
    #plot.set(ylim=(260, 285))
    #sns.despine(left=True, bottom=True)
    plt.savefig(results_folder + experiment + '_' + exploration + "_" + parameterValues + '_TEST.png')
    plt.close()




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
    plt.close()

    protocol2['indexes'] = protocol2.index
    protocol2.plot(x='indexes', y='comprForce', kind='scatter', figsize=(10, 5), fontsize=15, lw=4, c=protocol2['axis'].map(colors))
    #plt.ylim([-0.001, 0.01])
    plt.xlabel('Learning episodes', fontsize=15)
    plt.ylabel('comprForce', fontsize=15)
    plt.legend(loc='upper right')
    plt.savefig(results_folder+experiment + '_' + exploration + "_" + stop_epoch_name + '_protocol.png')
    plt.close()



if __name__ == '__main__':

    results_folder="results/experiment2/"
    lr=str(1e-05)
    gamma=str(0.95)
    epoch="20"
    experiment='2_final_fraction_cells'

    plotPerformanceCellsInside(results_folder=results_folder+"new_palacell_out_circles_"+lr+"_"+gamma+"/", data_file="data_to_save_at_epoch_"+epoch+".npy", experiment='1_final_n_cells', exploration='lr_gamma', parameterValues=lr+"_"+gamma)
    
    
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
            
