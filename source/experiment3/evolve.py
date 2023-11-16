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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import env.checks as checks
import pygad

base_outfolder = "../../results"

class Evolve:
    def __init__(self, env, simulation_duration, n_protocol_segments, num_generations, num_parents_mating, id):
        
        #environment instance
        self.env = env

        #simulation duration set up
        #TODO: make this setting more organic
        self.simulation_duration=simulation_duration
        self.env.max_iterations=self.simulation_duration

        # protocol temporal structure set up
        # n of protocol segments sets the n of epochs in the env
        self.n_protocol_segments=n_protocol_segments
        self.env.epochs=self.n_protocol_segments
        self.env.iters= abs(self.simulation_duration / self.n_protocol_segments)

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
        self.initial_population=self.generate_initial_protocols()
        self.fitness_func=self.fitness_func()
        self.on_generation=self.on_generation()

        #optimization process id
        self.id = id
    
    
    # fitness equals the final n of cells in a simulation
    # this function must simulate the environment using
    # the solution to stimulate it
    # and compute fitness getting the final n of cells from it
    def fitness_function(self, ga_instance, solution, sol_idx):

        # reset environment
        self.env.reset()

        # act on environment with solution as protocol
        # the step() function executes iters simulation steps
        # so it can sustain the entire simulation execution.
        # but uses the same actions along the entire simulation
        # let's use epochs within this function to handle temporal structure of protocol

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

        for j in range(starting_epoch,epochs):
            
            # j selects the protocol segment
            # for each segment an axis and a comprForce level
            axis=protocol[j][0]
            comprForce=protocol[j][1]

            #for iters simulation steps, administering these stimuli
            self.env.step([axis, comprForce])

        #getting the final n of cells at the end of the simulation
        solution_fitness = self.env.get_performance()

        return solution_fitness

    # pygad method to print stats at every generation
    def on_generation(self):
        print(f"Generation = {self.ga_instance.generations_completed}")
        print(f"Fitness    = {self.ga_instance.best_solution()[1]}")
    
    
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

    def evolve(self, save_every=10, verbose=True, recv=None, starting_epoch=0):
        
        #TODO: substitute tf logging with pyGAD logging
        #tf.get_logger().setLevel('ERROR')   

        #check for env integrity
        checks.check_env(self.env)

        '''
        setup the environment
        '''
        num_continue = self.env.num_continue
        num_discrete = self.env.num_discrete
        self.env.save_performance([])

        '''
        setup evolutive process
        '''
        # creating ga instance
        ga_instance = pygad.GA(num_generations=self.num_generations,
                       num_parents_mating=self.num_parents_mating,
                       #initial_population=self.initial_population,
                       num_genes=self.n_protocol_segments,
                       fitness_func=self.fitness_func,
                       on_generation=self.on_generation())
        
        self.ga_instance = ga_instance

        epochs = self.env.epochs
        iterations = self.env.iterations

        output_dir = self.env.output_dir+"_"+str(self.num_generations)+"_"+str(self.num_parents_mating)
        if not os.path.exists(base_outfolder):
            os.makedirs(base_outfolder)
        if not os.path.exists(base_outfolder+"/"+output_dir):
            os.makedirs(base_outfolder+"/"+output_dir)

        self.ga_instance.run()

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
