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

lr = 0.0001
gamma = 0.95

iters_list = [20, 50, 100]

base_outfolder = "../../results"

def parallel_train(testingMode=False):

    #set when loading previous state
    starting_epoch = 0

    envs = []
    trains = []

    for i, numIter in enumerate(iters_list):
        
        if testingMode:
            print("ENTERING TESTING MODE FOR", numIter)
            starting_epoch=70
            env = penv.PalacellEnv(iters=numIter, configuration_file='new_circles_iters_'+str(numIter)+"_"+str(lr)+'_'+str(gamma)+'.xml', output_file='chem__'+str(numIter)+"_"+str(lr)+'_'+str(gamma)+'-', output_dir='experiment2_iters/new_palacell_out_circles_iters'+str(numIter), max_iterations=3400, mode='circles', target=[200,250,80],lr=lr, gamma=gamma, starting_epoch=starting_epoch,
            preload_model_weights = True,
            preload_data_to_save = True,
            preload_performance = True,
            testingMode=testingMode)
        
            print("Env created", env)

            env.epochs = 101 
            current_env = str(lr)+'_'+str(gamma)

            env.preload_model_weights = base_outfolder+"/"+env.output_dir+'_'+current_env+"/model_at_epoch_"+str(starting_epoch)+".h5"
            env.preload_data_to_save = base_outfolder+"/"+env.output_dir+'_'+current_env+"/data_to_save_at_epoch_"+str(starting_epoch)
            env.preload_performance = base_outfolder+"/"+env.output_dir+'_'+current_env+"/performance_at_epoch_"+str(starting_epoch)
            
        else:
            starting_epoch=0
            env = penv.PalacellEnv(iters=numIter, configuration_file='new_circles_iters_'+str(numIter)+"_"+str(lr)+'_'+str(gamma)+'.xml', output_file='chem__'+str(numIter)+"_"+str(lr)+'_'+str(gamma)+'-', output_dir='experiment2_iters/new_palacell_out_circles_iters'+str(numIter), max_iterations=3400, mode='circles', target=[200,250,80],lr=lr, gamma=gamma, starting_epoch=starting_epoch)
                #preload_model_weights = True,
                #preload_data_to_save = True,
                #preload_performance = True
            #)
        
            env.epochs = 71

        #set the following line and one of the following two blocks when starting from a previous training
        #also uncomment the three 'preload' lines in the PalacellEnv constructor
        #the first block refers to any data file produced during the training
        #the second block refers to data files produced after the last possible epoch, if the previous training has been completed
        #current_env = str(lr)+'_'+str(gamma)

        #from starting_epoch
        #env.preload_model_weights = base_outfolder+"/"+env.output_dir+'_'+current_env+"/model_at_epoch_"+str(starting_epoch)+".h5"
        #env.preload_data_to_save = base_outfolder+"/"+env.output_dir+'_'+current_env+"/data_to_save_at_epoch_"+str(starting_epoch)
        #env.preload_performance = base_outfolder+"/"+env.output_dir+'_'+current_env+"/performance_at_epoch_"+str(starting_epoch)

        #after another training has been fully completed
        #env.preload_model_weights = base_outfolder+"/"+env.output_dir+'_'+current_env+"/last_model.h5"
        #env.preload_data_to_save = base_outfolder+"/"+env.output_dir+'_'+current_env+"/last_data_to_save"
        #env.preload_performance = base_outfolder+"/"+env.output_dir+'_'+current_env+"/last_performance"

        envs.append(env)

    processes = []
    senders = []
    for i in range(len(iters_list)):
        recv, send = Pipe()
        train = Train(envs[i], lr, gamma, i)
        proc = Process(target=train.train, args=[5, False, recv, starting_epoch, testingMode])
        proc.start()
        trains.append(train)
        processes.append(proc)
        senders.append(send)
        
    start_time = time.time()

    time.sleep(10)
    
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
    
    vdisplay = Xvfb()
    vdisplay.start()
    parallel_train(testingMode=True)
    vdisplay.stop()
