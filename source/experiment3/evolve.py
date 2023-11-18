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
        self.num_parents_mating=num_parents_mating
        #self.initial_population=self.generate_initial_protocols()
        #self.fitness_func=self.fitness_function()
        #self.on_generation=self.on_generation()
        self.sol_per_pop=sol_per_pop

        
        #optimization process id
        self.id = id
    
    #a function to support the fitness_func()
    #in simulating the administration of the solution
    #leveraging on the environment instance
    def administer_solution(self, solution):

        # reset environment
        self.env.reset()

        # act on environment with solution as protocol
        # the step() function executes iters simulation steps
        # we use epochs within this function to handle temporal structure of protocol
        # 1 epoch - 1 segment

        # we consider one epoch with iters simulation steps per protocol segment
        starting_epoch=0
        epochs=self.env.epochs
        
        # protocol segments as a list of genes in the solution
        # TODO: understand if pygad supports lists as genes
        # if not fall back on using only comprForce
        # more precisely, a list of lists
        # external list is segments
        # internal lists have two elements: Axis and comprForce
        protocol = solution


        #print("The solution is: ", protocol)
        
        for j in range(starting_epoch,epochs):
            
            # j selects the protocol segment
            # for each segment a comprForce level
            #TODO: integrate also axis
            #axis=protocol[j][0]
            comprForce=protocol[j]#[1]

            #for iters simulation steps, administering
            #-generated comprForce stimuli 
            #axis value is generated randomly for each segment
            #TODO: evolve also axis value

            axis=random.choice(['X','Y'])
           
            self.env.step([axis, comprForce])

        #getting the final n of cells at the end of the simulation
        nCells = self.env.get_performance()

        return nCells

    #a function to support the fitness_func()
    #in computing the fitness of the solution
    #leveraging on the returned value from simulation
    #can support simple passing, or be made more complex
    def computeFitness(self, nCells):

        #passing the final n of cells at the end of the simulation
        solution_fitness = nCells

        return solution_fitness
    
    # fitness equals the final n of cells in a simulation
    # this function must call functions to simulate the environment using
    # the solution to stimulate it
    # and compute fitness getting the final n of cells from functions to render the environment
    def fitness_func(self, ga_instance, solution, sol_idx):

        #use solution as a protocol and administer it to the env
        #get final n of cells at the end of the simulation
        nCells=self.administer_solution(solution=solution)

        solution_fitness=self.computeFitness(nCells=nCells)

        return solution_fitness

    # pygad method to print stats at every generation
    def on_generation(self, ga_instance):
        print(f"Generation = {self.ga_instance.generations_completed}")
        print(f"Fitness    = {self.ga_instance.best_solution()[1]}")
        
        #pyGAD instance saving
        self.ga_instance.save(filename=output_dir+str(self.ga_instance.generations_completed)+"_generation_ga_instance")
        
        #pyGAD solution fitness visualization and saving
        fitnesses=self.ga_instance.cal_pop_fitness()
        
        # CSV file to append the vector
        fitnessTrack = output_dir+"/fitness_track.csv"

        # Open the file in append mode and append the vector
        with open(fitnessTrack, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(fitnesses)




    
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

    def evolve(self, save_every=10, verbose=True, recv=None):
        
        #TODO: substitute tf logging with pyGAD logging

        #check for env integrity
        checks.check_env(self.env)
        
        #setup the environment savings
        self.env.save_performance([])
        
        output_dir = self.env.output_dir+"_"+str(self.num_generations)+"_"+str(self.num_parents_mating)
        
        if not os.path.exists(base_outfolder):
            os.makedirs(base_outfolder)
        if not os.path.exists(base_outfolder+"/"+output_dir):
            os.makedirs(base_outfolder+"/"+output_dir)

        #initializing ga instance
        ga_instance = pygad.GA(num_generations=self.num_generations,
                       num_parents_mating=self.num_parents_mating,
                       #initial_population=self.initial_population,
                       init_range_low=0,
                       init_range_high=10,
                       num_genes=self.n_protocol_segments,
                       sol_per_pop=self.sol_per_pop,
                       fitness_func=self.fitness_func,
                       on_generation=self.on_generation,
                       save_best_solutions=True,
                       save_solutions=True)

        ga_instance.run()

"""
    Function stubs to manage more protocol structural complexity in the future (or with other libraries)

    # defines stimuli types and ranges
    # returns dictionary of stimuli names (keys) and ranges (arrays)
    def define_stimuli(self, names, ranges):
        
        #TODO: generate stimuli ranges, link them to names
        stimuli = ""

        return stimuli

    # generates a pool of n initial protocols 
    # returns a list of arrays
    def generate_initial_protocols(self, n_protocols):

        #generate arrays with shape s x d and random values in relevant ranges
        #s -> stimuli (es: compression level, compression axis)
        #d -> duration (es: 3400 simulation steps)

        #TODO: for each stimulus in stimuli dict, generate a random value from range
        # for each simulation step

        initial_protocols = []
        for p in range(n_protocols):
            initial_protocols.append(self.generate_random_protocol)

        return initial_protocols
    
    # generates a protocol with random stimuli values
    # duration: int n of simulation steps for protocol duration
    # stimuli: dictionary of stimuli names (keys) and ranges (arrays)
    # returns a list of arrays
    def generate_random_protocol(self, duration, stimuli): - focusing on comprForce only for now

        #generate arrays with shape s x d and random values in relevant ranges
        #s -> stimuli (es: compression level, compression axis)
        #d -> duration (es: 3400 simulation steps)

        #TODO: for each stimulus in stimuli dict, generate a random value from range
        # for each simulation step

        random_protocol = ""

        return random_protocol

        """
