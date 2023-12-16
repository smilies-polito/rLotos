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
import time
import os
import pathlib
import re
import csv
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import env.checks as checks
import pygad

base_outfolder = "../../results"

class Evolve:
    def __init__(self, env, simulation_duration, n_protocol_segments, sol_per_pop, num_generations, num_parents_mating, id):
        
        #global env
        #self.env=env
        #environment instance
        self.env = env

        #simulation duration set up
        #TODO: make this setting more organic
        self.simulation_duration=simulation_duration
        #self.env.max_iterations=self.simulation_duration
        env.max_iterations=self.simulation_duration

        # protocol temporal structure set up
        # n of protocol segments sets the n of epochs in the env
        self.n_protocol_segments=n_protocol_segments
        #self.env.epochs=self.n_protocol_segments
        env.epochs=self.n_protocol_segments
        #self.env.iters= abs(self.simulation_duration / self.n_protocol_segments)
        env.iters= abs(self.simulation_duration / self.n_protocol_segments)

        #TODO: this can be used to assign different weights to different segments
        #for now let's just use the n of protocol segments to set num_genes
        #protocol will be a list of positive int valued 1
        #in fact these are constants multiplying explored variables
        #each value marks a protocol segment
        #function_inputs=[]
        #for s in self.n_protocol_segments:
        #    function_inputs.append(1)
        #self.function_inputs=function_inputs

        #ga hyperparameters from pyGAD
        self.num_generations=num_generations
        #self.initial_population=initial_population
        self.num_parents_mating=num_parents_mating
        #self.initial_population=self.generate_initial_protocols()
        #self.fitness_func=self.fitness_function()
        #self.on_generation=self.on_generation()
        self.sol_per_pop=sol_per_pop

        
        #optimization process id
        self.id = id

        print ("Process ", self.id, " initialized")
    
    #a function to support the fitness_func()
    #in simulating the administration of the solution
    #leveraging on the environment instance
    def administer_solution(self, solution, sol_idx):
        
        # protocol segments as a list of genes in the solution
        # TODO: understand if pygad supports lists as genes
        # if not fall back on using only comprForce
        # more precisely, a list of lists
        # external list is segments
        # internal lists have two elements: Axis and comprForce
        protocol = solution

        #each protocol is composed by two macro-parts: 
        #A - the setting of the initial cell position
        #B - the sequence of compr stimuli and axes for the n protocol segments

        # A - initial cell positioning
        # TODO parametrize on n of spatial dimensions
        xCoord=protocol[0]
        yCoord=protocol[1]
        coordinates=[xCoord, yCoord]

        # reset environment + pass coordinates
        self.env.reset()
        self.env.setPosition(coordinates)
        
        # act on environment with solution as protocol
        # the step() function executes iters simulation steps
        # we use epochs within this function to handle temporal structure of protocol
        # 1 epoch - 1 segment

        # we consider one epoch with iters simulation steps per protocol segment
        starting_epoch=0
        epochs=self.env.epochs
        

        print("Process ", self.id, " administering solution:", " ".join(str(e) for e in solution), "with ", len(protocol), "genes and ", epochs, "protocol segments")   

        #initialize cell increments list
        cellIncrements=[]

        #initialize normFrac increments list
        normFracIncrements=[]

        #initialize compression history list
        comprHistory=[]

        #each protocol is composed by two macro-parts: 
        #A - the setting of the initial cell position
        #B - the sequence of compr stimuli and axes for the n protocol segments


        self.env.configure(self.env.configuration, self.env.iters, export_step=self.env.iters, initialPath = "output/"+self.env.output_file+"_final_cell.vtp",
        finalPath = self.env.output_file, initial_position=np.array([xCoord, yCoord]))

        print("Process ", self.id, "solution ", sol_idx, "setting initial coordinates x: ", xCoord, "y: ", yCoord)

        
        # B - compression along simulation
        # each segment is composed by 2 genes
        # the first is for comprForce
        # the second for axis
        # at this point, the number of genes (len(protocol)) is twice the number of protocol segments (epochs)
        # iterating over protocol length, with a step of 2, to elaborate both genes in each segment
        # the range starts from the third gene, since the first two are the coordinates
        # TODO parametrize on n of spatial dimensions
        for j in range(2,len(protocol), 2):
            
            # j selects the protocol segment
            # for each segment a comprForce level
            # and an axis level (0 -> "X", 1 -> "Y"))
            comprForce=protocol[j]
            axis=protocol[j+1]
            if axis == 0:
                axis="X"
            elif axis == 1:
                axis = "Y"

            #for iters simulation steps, administering
            # - generated comprForce stimuli 
            # - generated axis value
            
            #compute nCells before step
            nCells_before=self.env.getCellNum()

            #compute fraction of inside cells before step
            normFrac_before=self.env.getNormFrac()

            env_actions, _, _, _, _, _, _ = self.env.step([axis, comprForce])

            print("Process ", self.id, "solution ", sol_idx, "protocol segment ", str(int(j/2)),"over", str(len(protocol)), "protocol segments, genes ",j, "and", str(j+1),   "\n Administering a compression stimulus of value ", env_actions[1], " on the ", env_actions[0], " axis for ", self.env.iters, " simulation steps")

            comprHistory.append(env_actions)

            #compute nCells after step
            nCells_after=self.env.getCellNum()

            #compute fraction of inside cells after step
            normFrac_after=self.env.getNormFrac()

            
            incrementNCells = nCells_after-nCells_before
            incrementNormFrac = normFrac_after - normFrac_before
            
            #append increment to cellIncrements list
            cellIncrements.append(incrementNCells)
            normFracIncrements.append(incrementNormFrac)

            #printing n of cells after protocol segment
            print("N of cells before and after protocol segment: ", nCells_before, nCells_after, "increment:", incrementNCells)
            #printing normFrac after protocol segment
            print("Normalized fraction of cells inside the target before and after protocol segment: ", normFrac_before, normFrac_after, "increment:", incrementNormFrac)

        #getting the final n of cells at the end of the simulation
        nCells = self.env.getCellNum()

        #getting the final Normalized fraction of cells inside the target at the end of the simulation
        NormFrac = self.env.getNormFrac()

        # save everything - solution, cell increments, solution fitness
        with open(base_outfolder+"/"+self.output_dir+"/output.csv", "a+") as f:
            f.write(" ,"+str(sol_idx)+","+str(xCoord)+","+str(yCoord)+";".join(str(sublist).replace(", ","|") for sublist in comprHistory)+","+";".join(str(i) for i in cellIncrements)+","+str(nCells)+","+str(NormFrac)+"\n")


        print("Process ", self.id, " solution ", sol_idx, ": protocol administration made ", nCells, "cells grow!", NormFrac, "is cells inside / n cells")

        return nCells, NormFrac

    #a function to support the fitness_func()
    #in computing the fitness of the solution
    #leveraging on the returned value from simulation
    #can support simple passing, or be made more complex
    def computeFitness(self, nCells, normFrac):

        #passing the final n of cells at the end of the simulation
        cells = nCells
        
        #passing the final norm frac of cells inside target at the end of the simulation
        solution_fitness = normFrac

        return solution_fitness
    
    # fitness equals the final n of cells in a simulation
    # this function must call functions to simulate the environment using
    # the solution to stimulate it
    # and compute fitness getting the final n of cells from functions to render the environment
    def fitness_func(self, ga_instance, solution, sol_idx):

        #use solution as a protocol and administer it to the env
        #get final n of cells at the end of the simulation
        nCells, normFrac=self.administer_solution(solution=solution, sol_idx=sol_idx)

        solution_fitness=self.computeFitness(nCells=nCells, normFrac=normFrac)
        
        print("Process ", self.id, " generation ", str(ga_instance.generations_completed), " solution ", sol_idx, " has fitness ", solution_fitness)

        return solution_fitness

    # pygad method to print stats at every generation
    def on_generation(self, ga_instance):
        print("***************************************************\n***************************************************\n**************** GENERATION COMPLETED *************\nPROCESS ", self.id, " GENERATIONS ", ga_instance.generations_completed, " BEST SOLUTION FITNESS ", ga_instance.best_solution()[1], "\n***************************************************\n***************************************************")

        #pyGAD instance saving
        ga_instance.save(filename=base_outfolder+"/"+self.output_dir+"/"+str(ga_instance.generations_completed)+"_generation_ga_instance")
        print("pyGAD instance saved at "+base_outfolder+"/"+self.output_dir+"/"+str(ga_instance.generations_completed)+"_generation_ga_instance")

        #pyGAD solution fitness visualization and saving
        fitnesses=ga_instance.cal_pop_fitness()
        
        # CSV file to append the vector
        fitnessTrack = base_outfolder+"/"+self.output_dir+"/fitness_track.csv"
        # saving solution fitnesses
        with open(fitnessTrack, 'a+') as f:
            f.write(str(ga_instance.generations_completed)+","+";".join(str(f) for f in fitnesses)+"\n")

        print("Solution fitness values saved at "+base_outfolder+"/"+self.output_dir+"/fitness_track.csv")
        
        #computing generation times
        generation_t_exec = time.time()
        generation_t_cpu = time.process_time()

        # saving times
        with open(base_outfolder + '/' + self.output_dir + '/timesLog.csv', 'a+') as f:                  
            f.write(str(ga_instance.generations_completed)+","+str(generation_t_exec)+','+str(generation_t_cpu)+"\n")
        
        # save everything - generation
        with open(base_outfolder+"/"+self.output_dir+"/output.csv", "a+") as f:
            f.write(str(ga_instance.generations_completed)+", , , , \n")


    
    def get_infos(self):
        strings = [""]
        strings.append("evolve id: "+str(self.id))
        strings.append("env infos: "+self.env._get_info())

        strings.append("population fitness infos: "+self.ga_instance.cal_pop_fitness())

        #returns:
        # best_solution: Best solution in the current population.
        # best_solution_fitness: Fitness value of the best solution.
        # best_match_idx: Index of the best solution in the current population.
        strings.append("best solution infos: "+self.ga_instance.best_solution())


        strings.append("")
        for s in strings:
            print(s)
        return strings

    def evolve(self, save_every=10, verbose=True, recv=None, restart=False,restart_epoch=None):

        #check for env integrity
        checks.check_env(self.env)
        
        #setup the environment savings
        self.env.save_performance([])
        
        self.output_dir = self.env.output_dir+"_"+str(self.env.iters)
        #self.output_dir = self.env.output_dir+"_"+str(self.sol_per_pop)+"_"+str(self.num_parents_mating)
        
        if not os.path.exists(base_outfolder):
            os.makedirs(base_outfolder)
        if not os.path.exists(base_outfolder+"/"+self.output_dir):
            os.makedirs(base_outfolder+"/"+self.output_dir)
        
        #computing start times
        start_t_exec = time.time()
        start_t_cpu = time.process_time()
        
        #setting up file to track fitnesses
        with open(base_outfolder+"/"+self.output_dir+"/fitness_track.csv", "a+") as f:
            #header has as many fields as the population numerosity
            f.write("Generations completed,Cells\n")#+","+str(n for n in range(self.sol_per_pop)))

        #setting up file to track times
        with open(base_outfolder + '/' + self.output_dir + '/timesLog.csv', 'a+') as f:
            f.write(str("Generations completed,Execution Time,CPU Time\n"))
        
        #setting up file to save everything - generation, solution, cell increments, solution fitness
        with open(base_outfolder+"/"+self.output_dir+"/output.csv", "a+") as f:
            f.write("Generations completed,Solution index,x,y,Solution,Cell increments,Final n cells,Final Inside Fraction\n")

        # setting the gene space for each gene
        gene_space=[]

        # setting gene spaces for A - initial positioning of cells
        # TODO parametrize on n of spatial dimensions
        gene_space.append({'low': 90.0, 'high': 310.0}) # x coordinate
        gene_space.append({'low': 90.0, 'high': 310.0}) # y coordinate

        #setting gene spaces for B - compression stimuli along simulation
        for s in range(self.n_protocol_segments):
            
            #for each protocol segment, appending two sublists:
            #the first for the comprForce stimulus range, floats from 0.0 to 10.0, as a dictionary of lower and upper bounds, and the step for range construction
            #the second for the compression axis, either 1 or 0, as int values, to be translated in "X" and "Y"
            gene_space.append({'low': 0.0, 'high': 5.0}) # generating gene space
            gene_space.append([0,1])

        
        # A - initial positioning of cells
        # it includes 2 genes, one for x and the other for y coordinates respectively
        # TODO parametrize on n of spatial dimensions
        genesA=2 # x and y coordinates

        # B - compression stimuli along simulation
        #setting n of genes as n_protocol_segments*2
        #since each segment has both comprForce and axis as genes
        genesB=self.n_protocol_segments*2

        num_genes=genesA+genesB

        if not restart:
            #initializing ga instance
            ga_instance = pygad.GA(num_generations=self.num_generations,
                        #init_range_low=0.0,
                        #init_range_high=5.0,
                        num_genes=num_genes,
                        num_parents_mating=self.num_parents_mating,
                        sol_per_pop=self.sol_per_pop,
                        fitness_func=self.fitness_func,
                        on_generation=self.on_generation,
                        #mutation_percent_genes=40, - defaulting to 10%
                        mutation_type="random",
                        mutation_by_replacement=True,
                        gene_space=gene_space)
                        #save_best_solutions=True)#,
                        #save_solutions=True)

        else:
            
            #   TODO - make the selection of the latest epoch ga instance automatic based on number in filename
            # alternative - save the last epoch with special name and overwrite at every epoch
            #ga_instances = [f for f in pathlib.Path(str(base_outfolder)+"/"+str(self.output_dir)+"/").glob("*.pkl")]
            #last_ga_instance=max(ga_instances, key=extract_number)

            # for now, inserting epoch manually launching one process at a time
            last_epoch=restart_epoch

            #load() requires to not put the .pkl extension in the filename
            ga_instance=pygad.load(str(base_outfolder)+"/"+str(self.output_dir)+"/"+str(last_epoch)+"_generation_ga_instance")

        ga_instance.run()

