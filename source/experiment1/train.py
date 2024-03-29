'''
Credits: Alberto Castrignanò
            - albertocastrignano2@gmail.com
            - https://www.linkedin.com/in/albertocastrignano2/
            - https://github.com/AlbInitPolito
'''

import numpy as np
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import env.checks as checks
from model import ActorCritic

base_outfolder = "../../results"

class Train:
    def __init__(self, env, lr, gamma, id):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.scores = []
        self.losses = []
        self.loss = 0
        self.epoch = 0
        self.id = id

    def get_infos(self):
        strings = [""]
        strings.append("train id: "+str(self.id))
        #strings.append("scores: "+str(self.scores))
        strings.append("losses: "+str(self.loss))
        strings.append("epoch: "+str(self.epoch))
        #strings.append("times: "+str(self.times))
        strings.append("env infos: "+self.env._get_info())

        strings.append("")
        for s in strings:
            print(s)
        return strings

    def train(self, save_every=10, verbose=True, recv=None, starting_epoch=0, testingMode=False):
        lr = self.lr
        gamma = self.gamma
        #use cpu if problems while testing on laptop (decomment following lines)
        #tf.config.set_visible_devices([], 'GPU')
        tf.get_logger().setLevel('ERROR')
        
        num_continue = self.env.num_continue
        num_discrete = self.env.num_discrete
        range_continue = self.env.range_continue
        dim_discrete = self.env.dim_discrete
        model = ActorCritic(num_continue=num_continue,num_discrete=num_discrete,range_continue=range_continue,dim_discrete=dim_discrete)
        self.model = model

        #check for env integrity
        checks.check_env(self.env)

        '''
        setup the environment
        '''
        num_continue = self.env.num_continue
        num_discrete = self.env.num_discrete
        self.env.save_performance([])

        '''
        setup training
        '''
        GAMMA = gamma
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        epochs = self.env.epochs
        iterations = self.env.iterations

        output_dir = self.env.output_dir+str(lr)+"_"+str(gamma)

        # creating testing subfolder for testing results
        if testingMode:
            print("TESTING MODE SET UP")
            output_dir = self.env.output_dir+"_"+str(lr)+"_"+str(gamma)+"/testing"
            print("Checking or creating folder to store testing results at: ", output_dir)
            if not os.path.exists(base_outfolder):
                os.makedirs(base_outfolder)
            if not os.path.exists(base_outfolder+"/"+output_dir):
                os.makedirs(base_outfolder+"/"+output_dir)
                print("created folder at "+base_outfolder+"/"+output_dir)
        else:
            if not os.path.exists(base_outfolder):
                os.makedirs(base_outfolder)
            if not os.path.exists(base_outfolder+"/"+output_dir):
                os.makedirs(base_outfolder+"/"+output_dir)

        self.model.build((1,self.env.width,self.env.height,self.env.channels))
        self.model.summary()

        if self.env.preload_model_weights:
            model.load_weights(self.env.preload_model_weights)
        if self.env.preload_losses:
            self.losses = np.load(self.env.preload_losses+".npy").tolist()
            self.losses.append(0)
        if self.env.preload_performance:
            performance_indexes = np.load(self.env.preload_performance+".npy", allow_pickle=True)
            self.env.load_performance(performance_indexes)
        if self.env.preload_data_to_save:
            self.env.load_data_to_save(np.load(self.env.preload_data_to_save+".npy", allow_pickle=True).item())

        print("GO: ", self.id)

        if starting_epoch>0:
            starting_epoch = starting_epoch+1
        for j in range (starting_epoch,epochs):
           
            '''
            compute epoch start execution and CPU times
            '''

            start_t_exec = time.time()
            start_t_cpu = time.process_time()


            if recv!=None and recv.poll():
                msg = recv.recv()
                if msg=='info':
                    self.get_infos()
            elapsed_time = time.time()
            self.epoch = j
            with tf.GradientTape() as tape:
                done = False

                rewards = []
                values = []
                log_probs = []
                discrete_log_probs = []

                observation = self.env.reset()
                observation = observation/255
                observation = tf.convert_to_tensor(observation)
                for iter in range(iterations):
                    if recv!=None and recv.poll():
                        msg = recv.recv()
                        if msg=='info':
                            self.get_infos()
                    
                    if testingMode:
                        print("TESTING MODE")
                        print("Testing epoch", j)
                        print("Iteration ", iter, "of", iterations, "total iterations")
                        
                        self.get_infos()
                    
                    '''
                    obtain actions and generate log probs
                    '''
                    
                    '''
                    compute action computation time
                    '''

                    action_start_t_exec = time.time()
                    action_start_t_cpu = time.process_time()


                    
                    
                    discrete_actions, continue_actions, normals, value = self.model(observation)
                    
                    # print("CONTINUE ACTION GENERATED: ")
                    # tf.print(continue_actions)


                    values.append(value[0][0])

                    #discrete actions
                    d_acts = []
                    if discrete_actions:
                        for da in discrete_actions:
                            probs = da[0].numpy().astype('float64')
                            action = np.random.choice(len(da[0]), size=1, p=(probs/sum(probs))) #obtain a random action based on the probability given by each discrete action
                            discrete_log_prob = tf.math.log(da[0][action[0]]) #log of probability of given action
                            discrete_log_probs.append(discrete_log_prob)
                            d_acts.append(action[0])
                    #continue actions
                    if continue_actions:
                        temp_cont = []
                        for (i,nd) in enumerate(normals):
                            log_prob = nd.log_prob(continue_actions[i]) #log of probability of given action
                            log_probs.append(log_prob)
                            temp_cont.append(tf.clip_by_value(continue_actions[i],self.env.range_continue[i][0],self.env.range_continue[i][1]))
                        
                        continue_actions = tf.convert_to_tensor(temp_cont)
                    
                    # print("CONTINUE ACTION GENERATED AFTER CLIPPING: ")
                    tf.print(continue_actions)
                    
                    '''
                    compute action generation total times
                    '''

                    action_stop_t_exec = time.time()
                    action_stop_t_cpu = time.process_time()

                    action_exec_time = action_stop_t_exec - action_start_t_exec
                    action_cpu_time = action_stop_t_cpu - action_start_t_cpu


                    '''
                    compute times to act on environment
                    '''
                    env_act_start_t_exec = time.time()
                    env_act_start_t_cpu = time.process_time()

                    '''
                    act on the environment
                    '''
                    observation, reward, done, info = self.env.step(self.env.adapt_actions(d_acts, continue_actions))
                    rewards.append(reward)
                    observation = observation/255
                    observation = tf.convert_to_tensor(observation)

                    if done:
                        break

                    #if iter%10==0 and verbose:
                    #if iter%50==0:
                    #    print("Iteration: ", iter)
                    #    print("Elapsed: ",str(time.time()-elapsed_time))
                    #    self.get_infos()
                
                '''
                compute partial times
                '''
                env_act_stop_t_exec = time.time()
                env_act_stop_t_cpu = time.process_time()

                env_act_exec_time = env_act_stop_t_exec - env_act_start_t_exec
                env_act_cpu_time = env_act_stop_t_cpu - env_act_start_t_cpu


                '''
                partial times for learning
                '''
                learning_start_t_exec = time.time()
                learning_start_t_cpu = time.process_time()


                '''
                compute loss and backpropagte
                '''
                #compute Q-values
                #_, _, _, Qval = model(last_observation)
                Qval = 0
                Qvals = np.zeros_like(values)
                for t in reversed(range(len(rewards))):
                    Qval = rewards[t] + GAMMA * Qval
                    Qvals[t] = Qval

                ##transform values, Qvals  into keras tensors   
                Qvals = tf.convert_to_tensor(Qvals)
                values = tf.convert_to_tensor(values)

                #compute advantage
                advantage = Qvals - values #ADVANTAGE IN TAKING ACTION A WRT ACTIONS IN THAT STATE (removes more noise)
                
                #compute actor loss
                if num_continue>0:
                    log_probs = tf.convert_to_tensor(log_probs)
                    actor_continue_loss = 0
                    for i in range(num_continue):
                        temp_log_probs = [-log_probs[j] for j in range(len(log_probs)) if (j+i)%num_continue==0]
                        actor_continue_loss += tf.math.reduce_mean(temp_log_probs*advantage)
                if num_discrete>0:
                    discrete_log_probs = tf.convert_to_tensor(discrete_log_probs)
                    actor_discrete_loss = tf.math.reduce_mean([-discrete_log_probs[i]*advantage[int(i/num_discrete)] for i in range(len(discrete_log_probs))])
                
                #compute critic loss and sum up everything
                critic_loss = 0.5 * tf.math.reduce_mean(advantage**2) ##MEAN SQUARE ERROR
                ac_loss = critic_loss
                if num_continue>0:
                    ac_loss += actor_continue_loss
                if num_discrete>0:
                    ac_loss += actor_discrete_loss
                ac_loss = tf.convert_to_tensor(ac_loss)

                #save model if it has better performance before changing it through backpropagation
                # or in case of model testing (testingMode=True)
                if self.env.check_performance([Qvals[0].numpy(),time.time()-elapsed_time,ac_loss,j]) or testingMode:
                    self.model.save_weights(base_outfolder+"/"+output_dir+"/model_at_epoch_"+str(j)+"(best).h5") 
                    np.save(base_outfolder+"/"+output_dir+"/data_to_save_at_epoch_"+str(j)+"(best)",self.env.data_to_save())
                    np.save(base_outfolder+"/"+output_dir+"/performance_at_epoch_"+str(j)+"(best)",self.env.get_performance())
                    np.save(base_outfolder+"/"+output_dir+"/losses_at_epoch_"+str(j)+"(best)",self.losses)

                
                if not testingMode:
                    #compute gradients and backpropagate
                    grads = tape.gradient(ac_loss, self.model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


                '''
                learning partial times
                '''

                learning_stop_t_exec = time.time()
                learning_stop_t_cpu = time.process_time()

                learning_exec_time = learning_stop_t_exec - learning_start_t_exec
                learning_cpu_time = learning_stop_t_cpu - learning_start_t_cpu
               
                '''
                compute epoch stop execution and CPU times
                '''
                stop_t_exec = time.time()
                stop_t_cpu = time.process_time()


                '''
                compute epoch execution and cpu time durations
                '''
                exec_time = stop_t_exec - start_t_exec
                cpu_time = stop_t_cpu - start_t_cpu
                
                '''
                save execution and cpu times
                '''
                with open(base_outfolder + '/' + output_dir + '/timesLog.csv', 'a+') as f:
                    
                    f.write(str(j)+'_action_generation,'+str(action_exec_time)+','+str(action_cpu_time)+'\n')
                    f.write(str(j)+'_act_on_env,'+str(env_act_exec_time)+','+str(env_act_cpu_time)+'\n')
                    f.write(str(j)+'_act_on_env,'+str(learning_exec_time)+','+str(learning_cpu_time)+'\n')
                    f.write(str(j)+','+str(exec_time)+','+str(cpu_time)+'\n')

                '''
                print epoch stats and save weights, scores, observations
                '''
                #TODO: ONLY FOR TESTING, REMOVE AFTER:
                #print("train id: ", self.id)
                #print("epoch: ", j, ", loss: ", ac_loss.numpy(), " lr: ", lr, " gamma: ", gamma)
                #print("Elapsed epoch time: ",str(time.time()-elapsed_time))
                self.losses.append(ac_loss)
                self.env.save_performance([Qvals[0].numpy(),time.time()-elapsed_time,ac_loss,j])
                if verbose:
                    print(len([i for i in grads if i==None]))
                #inds = [i for (i,j) in enumerate(grads) if j==None]
                #for i in inds:
                #    print(model.trainable_variables)
        
            '''
            save weights, scores, observations
            '''
            if j%save_every==0:
                self.model.save_weights(base_outfolder+"/"+output_dir+"/model_at_epoch_"+str(j)+".h5")
                np.save(base_outfolder+"/"+output_dir+"/data_to_save_at_epoch_"+str(j),self.env.data_to_save())
                np.save(base_outfolder+"/"+output_dir+"/performance_at_epoch_"+str(j),self.env.get_performance())
                np.save(base_outfolder+"/"+output_dir+"/losses_at_epoch_"+str(j),self.losses)

        self.model.save_weights(base_outfolder+"/"+output_dir+"/last_model.h5")
        np.save(base_outfolder+"/"+output_dir+"/full_losses",self.losses)
        np.save(base_outfolder+"/"+output_dir+"/last_data_to_save",self.env.data_to_save())
        np.save(base_outfolder+"/"+output_dir+"/last_performance",self.env.get_performance())
