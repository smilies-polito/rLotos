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
from xvfbwrapper import Xvfb

lr_list = [0.001, 0.0001, 0.00001]
gamma_list = [0.99, 0.95]

base_outfolder = "../../results"

def parallel_train():

    #set when loading previous state
    starting_epoch = 0

    envs = []
    trains = []

    combs = itertools.product(lr_list, gamma_list)
    for i, (lr, gamma) in enumerate(combs):

        env = penv.PalacellEnv(iters=20, configuration_file='compr_'+str(lr)+'_'+str(gamma)+'.xml', output_file='chem_'+str(lr)+'_'+str(gamma)+'-',
            output_dir='experiment1.1/new_palacell_out', max_iterations=3400,
            )
        #env.epochs = 71
        env.epochs = 141

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

    combs = itertools.product(lr_list, gamma_list)
    for i, (lr, gamma) in enumerate(combs):
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

def single_train(idx, iters, starting_epoch, max_epochs, lr=0.0001, gamma = 0.99, testingMode=False):

    #set when loading previous epoch
    starting_epoch = starting_epoch

    env = penv.PalacellEnv(iters=iters, configuration_file='compr_'+str(lr)+'_'+str(gamma)+'.xml', output_file='chem_'+str(lr)+'_'+str(gamma)+'-',
            output_dir='experiment1.1/new_palacell_out', max_iterations=3400, id=idx)
    
    #max epochs
    env.epochs = max_epochs

    #set the following line and one of the following two blocks when starting from a previous training
    #the first block refers to any data file produced during the training
    #the second block refers to data files produced after the last possible epoch, if the previous training has been completed
    #current_env = str(lr)+'_'+str(gamma)
    current_env = 'iters_'+str(iters)+'_'+str(lr)+'_'+str(gamma)

    #set to load previous state
    #env.preload_model_weights = base_outfolder+"/"+env.output_dir+'_'+current_env+"/model_at_epoch_"+str(starting_epoch)+".h5"
    #env.preload_data_to_save = base_outfolder+"/"+env.output_dir+'_'+current_env+"/data_to_save_at_epoch_"+str(starting_epoch)
    #env.preload_performance = base_outfolder+"/"+env.output_dir+'_'+current_env+"/performance_at_epoch_"+str(starting_epoch)

    env.preload_model_weights = base_outfolder+"/"+env.output_dir+'_'+current_env+"/last_model.h5"
    env.preload_data_to_save = base_outfolder+"/"+env.output_dir+'_'+current_env+"/last_data_to_save"
    env.preload_performance = base_outfolder+"/"+env.output_dir+'_'+current_env+"/last_performance"

    train = Train(env, lr, gamma, idx)
    train.train(5, False, recv=None, starting_epoch=starting_epoch, testingMode=testingMode)
    



if __name__=='__main__':
    
    vdisplay = Xvfb()
    vdisplay.start()
    parallel_train()
    vdisplay.stop()
