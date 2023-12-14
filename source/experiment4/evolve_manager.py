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


import time
import env.PalacellEnv as penv
from evolve import Evolve
import itertools
from multiprocessing import Process
from multiprocessing import Pipe
from xvfbwrapper import Xvfb

#Values of initial population numerosity are set based on values from the documentation
#number of parents mating are chose to be lower or equal to the total populations generating different degrees of elitism
#protocol segment lengths are chosen from the numIter values explored in exp1.2

initial_population_list = [8]
protocol_segment_lengths_list = [20, 50, 100, 200]

num_parents_mating=4

base_outfolder = "../../results"

def parallel_evolve():

    envs = []
    evolutions = []

    combs = itertools.product(initial_population_list, protocol_segment_lengths_list)
    for i, (initial_population, protocol_segment_length) in enumerate(combs):

        # iters indicates the total simulation length required
        # max iterations the maximum simulation length accepted
        # (in this case it is not required to specify it but let's set them coherently)
        env = penv.PalacellEnv(iters=20, configuration_file='compr_'+str(initial_population)+'_'+str(protocol_segment_length)+'.xml', output_file='chem_'+str(initial_population)+'_'+str(protocol_segment_length)+'-', output_dir='experiment4/new_palacell_out', max_iterations=2500, mode='circles', target=[200,250,20])   
         
        print("Creating environment ", i, " with ", initial_population, " solutions per population, and ", protocol_segment_length, " protocol segment length.")

        envs.append(env)

    processes = []
    senders = []

    combs = itertools.product(initial_population_list, protocol_segment_lengths_list)
    for i, (initial_population, protocol_segment_length) in enumerate(combs):
        recv, send = Pipe()
        
        #setting simulation duration to 3400 time steps
        #protocol segments lasting n iters
        # be careful, with percentage of mutated genes of 10% and 10 genes (eg, sim duration 500, protocol segments 5) it yields a warning that 0 genes get mutated 
        # selecting parameters to have more than 10 genes
        #and in accordance with the 5 iter per epoch in exp1
        
        #set simulation duration
        simulation_duration=2500
        n_protocol_segments=int(simulation_duration/protocol_segment_length)

        evolve = Evolve(envs[i], simulation_duration=simulation_duration, n_protocol_segments=n_protocol_segments, sol_per_pop=initial_population, num_generations=100, num_parents_mating=num_parents_mating, id=i)
        print("Launching evolution process ", i, " with ", initial_population, " solutions per population, and ", protocol_segment_length, " protocol segment length.")   
        proc = Process(target=evolve.evolve, args=[5, True, recv])
        proc.start()
        evolutions.append(evolve)
        processes.append(proc)
        senders.append(send)
        
    start_time = time.time()
    
    time.sleep(6)

    while True:
        print("Select a number between 0 and ",len(evolutions)-1," to get infos: ")
        try:
            ind = input()
            if ind=="exit":
                for proc in processes:
                    proc.terminate()
                    proc.kill()
                    #print("exit")
                    exit()
            elif ind=="time":
                print(str(time.time()-start_time))
            else:
                senders[int(ind)].send('info')
        except Exception as e:
            print("insert a valid index!")
            print(e)

def single_evolve(idx, protocol_segment_length, initial_population, num_parents_mating, num_generations=100, simulation_duration=3400, restart=False, restart_epoch=None):

    env = penv.PalacellEnv(iters=3400, configuration_file='compr_'+str(initial_population)+'_'+str(protocol_segment_length)+'.xml', output_file='chem_'+str(initial_population)+'_'+str(protocol_segment_length)+'-', output_dir='experiment3/new_palacell_out', max_iterations=3400)   
         
    print("Creating environment ", idx, " with ", initial_population, " solutions per population, and ", protocol_segment_length, " protocol segment length.")

    #set simulation duration
    simulation_duration=simulation_duration
    n_protocol_segments=int(simulation_duration/protocol_segment_length)
    
    evolve = Evolve(env, simulation_duration=simulation_duration, n_protocol_segments=n_protocol_segments, sol_per_pop=initial_population, num_generations=num_generations, num_parents_mating=num_parents_mating, id=idx)
    print("Launching evolution process ", idx, " with ", initial_population, " solutions per population, and ", protocol_segment_length, " protocol segment length.")
    
    evolve.evolve(save_every=5, verbose=True, recv=None, restart=restart, restart_epoch=restart_epoch)  


if __name__=='__main__':
    
    vdisplay = Xvfb()
    vdisplay.start()
    parallel_evolve()
    
    #relaunching a process with id 0, protocol_segment_length 200, initial_population 8, num_generations 40 (since it restarts from 60), simulation_duration=3400,restart=True, restart_epoch=60
    #single_evolve(idx=0, protocol_segment_length=200, initial_population=8, num_parents_mating=4, num_generations=40, simulation_duration=3400, restart=True, restart_epoch=60)
    #relaunching a process with id 1, protocol_segment_length 100, initial_population 8, num_generations 40 (since it restarts from 60), simulation_duration=3400,restart=True, restart_epoch=60
    #single_evolve(idx=1, protocol_segment_length=100, initial_population=8, num_parents_mating=4, num_generations=40, simulation_duration=3400, restart=True, restart_epoch=60)
    #relaunching a process with id 2, protocol_segment_length 20, initial_population 8, num_generations 87 (since it restarts from 13), simulation_duration=3400,restart=True, restart_epoch=13
    #single_evolve(idx=2, protocol_segment_length=20, initial_population=8, num_parents_mating=4, num_generations=87, simulation_duration=3400, restart=True, restart_epoch=13)
    vdisplay.stop()
