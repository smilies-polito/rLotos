import pygad
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

#data=pygad.load("/Users/bardini/Documents/SMILIES Projects/rlotos/source/experiment3/9_generation_ga_instance")
#print(data.plot_new_solution_rate())

def plotTimes(iters):

    path="../../results/experiment3/new_palacell_out_"+str(iters)+".0/"
    
    df = pd.read_csv(path+"timesLog.csv")

    #Subtract value of first call to start from 0
    df["Execution Time"]=df["Execution Time"]-df["Execution Time"][0]

    df["Execution time increments"]=df["Execution Time"]
    #populate new col with Execution time increments in minutes
    for i in range(len(df)-1, 0, -1):
        if i > 0:
            df["Execution time increments"].iloc[i]= df["Execution Time"][i]-df["Execution Time"].iloc[i-1]
    
    df["Execution time increments"].iloc[0]= df["Execution time increments"].iloc[1:].mean()

    #do the same with CPU times
    df["CPU time increments"]=df["CPU Time"]
    #populate new col with Execution time increments in minutes
    for i in range(len(df)-1, 0, -1):
        if i > 0:
            df["CPU time increments"].iloc[i]= df["CPU Time"].iloc[i]-df["CPU Time"].iloc[i-1]
    
    df["CPU time increments"][0]= df["CPU time increments"].iloc[1:].mean()

    #from seconds to minutes
    df["Execution Time"]=df["Execution Time"]/60
    df["CPU Time"]=df["CPU Time"]/60
    df["Execution time increments"]=df["Execution time increments"]/60
    df["CPU time increments"]=df["CPU time increments"]/60

    #from minutes to hours
    df["Execution Time"]=df["Execution Time"]/60
    df["CPU Time"]=df["CPU Time"]/60
    df["Execution time increments"]=df["Execution time increments"]/60
    df["CPU time increments"]=df["CPU time increments"]/60


    fig = plt.figure()

    df.plot.bar(x = "Generations completed", y = "Execution time increments")
    plt.savefig(path+"Execution_time_increments_vs_gen.png")

    df.plot.line(x = "Generations completed", y = "Execution Time")
    plt.savefig(path+"execution_time_vs_gen.png")

    df.plot.line(x = "Execution Time", y = "Generations completed")
    plt.savefig(path+"gen_vs_execution_time.png")

    df.plot.bar(x = "Generations completed", y = "CPU time increments")
    plt.savefig(path+"CPU_time_increments_vs_gen.png")

    df.plot.line(x = "Generations completed", y = "CPU Time")
    plt.savefig(path+"CPU_time_vs_gen.png")

    df.plot.line(x = "CPU Time", y = "Generations completed")
    plt.savefig(path+"gen_vs_CPU_time.png")
    
    plt.close()

def plotFitness(iters):

    path="../../results/experiment3/new_palacell_out_"+str(iters)+".0/"
    df = pd.read_csv(path+"fitnessTrack.csv")
    df = df.reset_index()
    print(df.columns)

    df["Average fitness"]=np.zeros(len(df))
    df["Max fitness"]=np.zeros(len(df))

    print(df.columns)

    for i in range(len(df)):

        df["Average fitness"].iloc[i] = sum([int(f) for f in df["Generations completed"].iloc[i].split(";")])/len([int(f) for f in df["Generations completed"].iloc[i].split(";")])
        df["Max fitness"].iloc[i] = int(max(df["Generations completed"].iloc[i].split(";")))

    fig = plt.figure()

    df.plot.line(x = "index", y = "Average fitness")
    plt.savefig(path+"Average_fitness_vs_gen.png")

    df.plot.line(x = "index", y = "Max fitness")
    plt.savefig(path+"Max_fitness_vs_gen.png")

    plt.close()



plotTimes(200)
plotTimes(100)
plotTimes(20)

plotFitness(200)
plotFitness(100)
plotFitness(20)