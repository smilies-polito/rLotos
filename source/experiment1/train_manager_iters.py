'''
Credits: Alberto Castrignanò
            - albertocastrignano2@gmail.com
            - https://www.linkedin.com/in/albertocastrignano2/
            - https://github.com/AlbInitPolito
'''

import time
import env.PalacellEnv as penv
from train import Train
import itertools
from multiprocessing import Process
from multiprocessing import Pipe

iters_list = [5,10,30,40,50,60,70,80,100,200]
lr = 0.0001
gamma = 0.99

base_outfolder = "../../results"

def parallel_train():

    #set when loading previous state
    starting_epoch = 0

    envs = []
    trains = []

    for i, iters in enumerate(iters_list):

        env = penv.PalacellEnv(iters=iters, configuration_file='compr_'+str(lr)+'_'+str(gamma)+'_iters_'+str(iters)+'.xml', output_file='chem_'+str(lr)+'_'+str(gamma)+'_'+str(iters)+'-',
            output_dir='experiment1.2/new_palacell_out_iters_'+str(iters), max_iterations=3400,
            lr=lr, gamma=gamma)
        env.epochs = 71

        #set the following line and one of the following two blocks when starting from a previous training
        #the first block refers to any data file produced during the training
        #the second block refers to data files produced after the last possible epoch, if the previous training has been completed
        #current_env = str(lr)+'_'+str(gamma)

        #set to load previous state
        #env.preload_model_weights = base_outfolder+"/"+env.output_dir+'_'+current_env+"/model_at_epoch_"+str(starting_epoch)+".h5"
        #env.preload_data_to_save = base_outfolder+"/"+env.output_dir+'_'+current_env+"/data_to_save_at_epoch_"+str(starting_epoch)
        #env.preload_performance = base_outfolder+"/"+env.output_dir+'_'+current_env+"/performance_at_epoch_"+str(starting_epoch)

        #env.preload_model_weights = base_outfolder+"/"+env.output_dir+'_'+current_env+"/last_model.h5"
        #env.preload_data_to_save = base_outfolder+"/"+env.output_dir+'_'+current_env+"/last_data_to_save"
        #env.preload_performance = base_outfolder+"/"+env.output_dir+'_'+current_env+"/last_performance"

        envs.append(env)

    processes = []
    senders = []

    for i in range(len(iters_list)):
        recv, send = Pipe()
        train = Train(envs[i], lr, gamma, i)
        proc = Process(target=train.train, args=[5, False, recv, starting_epoch])
        proc.start()
        trains.append(train)
        processes.append(proc)
        senders.append(send)
        
    start_time = time.time()
    
    time.sleep(6)

    while True:
        print("Select a number between 0 and ",len(trains)-1," to get infos: ")
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
    parallel_train()
