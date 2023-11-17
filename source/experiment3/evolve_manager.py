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
#from xvfbwrapper import Xvfb

#TODO: set relevant values
initial_population_list = [5, 10, 100]
num_parents_mating_list = [2, 5, 10]

base_outfolder = "../../results"

def parallel_evolve():

    #TODO: set relevant values
    initial_population_list = [5, 10, 100]
    num_parents_mating_list = [2, 5, 10]

    #set when loading previous state
    starting_epoch = 0

    envs = []
    evolutions = []

    combs = itertools.product(initial_population_list, num_parents_mating_list)
    for i, (initial_population, num_parents_mating) in enumerate(combs):

        # iters indicates the total simulation length required
        # max iterations the maximum simulation length accepted
        # (in this case it is not required to specify it but let's set them coherently)
        env = penv.PalacellEnv(iters=3400, configuration_file='compr_'+str(initial_population)+'_'+str(num_parents_mating)+'.xml', output_file='chem_'+str(initial_population)+'_'+str(num_parents_mating)+'-',
            output_dir='experiment3/new_palacell_out', max_iterations=3400)
        
        

        envs.append(env)

    processes = []
    senders = []

    

    combs = itertools.product(initial_population_list, num_parents_mating_list)
    for i, (initial_population, num_parents_mating) in enumerate(combs):
        recv, send = Pipe()
        
        #setting simulation duration to 500 time steps
        #protocol segments 5
        evolve = Evolve(envs[i], simulation_duration=500, n_protocol_segments=5, num_generations=10, num_parents_mating=num_parents_mating, id=i)
        proc = Process(target=evolve.evolve, args=[5, True, recv, starting_epoch])
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

if __name__=='__main__':
    
    #vdisplay = Xvfb()
    #vdisplay.start()
    parallel_evolve()
    #vdisplay.stop()
