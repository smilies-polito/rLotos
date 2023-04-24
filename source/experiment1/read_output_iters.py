'''
Credits: Alberto Castrignanò
            - albertocastrignano2@gmail.com
            - https://www.linkedin.com/in/albertocastrignano2/
            - https://github.com/AlbInitPolito
'''

import numpy as np
import matplotlib.pyplot as plt

####################################################################################
################################# LOAD OUTPUT DATA #################################
####################################################################################

# lr=0.0001 gamma=0.99 are fixed here
data_40 = np.load("../../results/experiment1.2/new_palacell_out_iters_40_0.0001_0.99/data_to_save_at_epoch_70.npy", allow_pickle=True).item()
data_50 = np.load("../../results/experiment1.2/new_palacell_out_iters_50_0.0001_0.99/data_to_save_at_epoch_70.npy", allow_pickle=True).item()
data_20 = np.load("../../results/experiment1.1/new_palacell_out_0.0001_0.99/data_to_save_at_epoch_70.npy", allow_pickle=True).item()
data = [data_40, data_50, data_20]

####################################################################################
################################ OBTAIN BEST RESULT ################################
####################################################################################

print("lr: 0.0001 gamma: 0.99 iters: 40 max cells: ", max(data_40['cell_numbers']))
print("lr: 0.0001 gamma: 0.99 iters: 50 max cells: ", max(data_50['cell_numbers']))
print("lr: 0.0001 gamma: 0.99 iters: 20 max cells: ", max(data_20['cell_numbers']))

# find best epoch
best_configuration = np.array(data_40['cell_numbers'])
best_epoch = np.where(best_configuration==max(best_configuration))[0][0] # should be 11

####################################################################################
################################# GENERATE FIGURES #################################
####################################################################################

x = list(range(71)) # x-axis

# plot the final number of cells for each hyperparameters configuration
plt.rcParams['font.size'] = '22'
plt.plot(x, data_40['cell_numbers'], label="iters=40")
plt.plot(x, data_50['cell_numbers'], label="iters=50")
plt.plot(x, data_20['cell_numbers'], label="iters=20")
plt.xlabel('epoch')
plt.ylabel('final number of cells')
plt.legend()
plt.show()

# compute mean and var for 6 windows:
# epochs 0-20, 10-30, 20-40, 30-50, 40-60, 50-70
means = []
vars = []
for _ in range(6):
    means.append([])
    vars.append([])
for i in range(6):
    for j in range(3):
        window = np.array(data[j]['cell_numbers'][10*i:10*i+20])
        means[j].append(np.mean(window))
        vars[j].append(np.var(window))

# plot mean of the 6 windows
x = list(range(6))
plt.rcParams['font.size'] = '22'
plt.xlabel('epoch')
plt.ylabel('mean of final number of cells')
plt.plot(x, means[0], label="iters=40")
plt.plot(x, means[1], label="iters=50")
plt.plot(x, means[2], label="iters=20")
plt.legend()
plt.show()

# plot var of the 6 windows
def do():
    plt.rcParams['font.size'] = '22'
    plt.xlabel('epoch')
    plt.ylabel('variance of final number of cells')
    plt.plot(x, vars[0], label="iters=40")
    plt.plot(x, vars[1], label="iters=50")
    plt.plot(x, vars[2], label="iters=20")
do()
plt.show()
do()
plt.legend()
plt.show()

# retrieve values of biofabrication protocol for epoch 6 and for best epoch (should be epoch 16)
#print(best_epoch)
protocol_epoch_0 = np.array(data_40['compr_history'][0])
protocol_best_epoch = np.array(data_40['compr_history'][best_epoch])

#use different colors when compression force is applied on X (red) and Y (blue) axes
iters = len(protocol_epoch_0)
lX = [i for i in range(iters) if protocol_epoch_0[i,0]=='X']
X = [float(p) for i,p in enumerate(protocol_epoch_0[:,1]) if protocol_epoch_0[i,0]=='X']
lY = [i for i in range(iters) if protocol_epoch_0[i,0]=='Y']
Y = [float(p) for i,p in enumerate(protocol_epoch_0[:,1]) if protocol_epoch_0[i,0]=='Y']
plt.rcParams['font.size'] = '22'
plt.scatter(lX, X, color='red', label='X axis')
plt.scatter(lY, Y, color='blue', label='Y axis')
plt.xlabel('iteration')
plt.ylabel('compression force')
plt.legend()
plt.show()

#use different colors when compression force is applied on X (red) and Y (blue) axes
iters = len(protocol_best_epoch)
lX = [i for i in range(iters) if protocol_best_epoch[i,0]=='X']
X = [float(p) for i,p in enumerate(protocol_best_epoch[:,1]) if protocol_best_epoch[i,0]=='X']
lY = [i for i in range(iters) if protocol_best_epoch[i,0]=='Y']
Y = [float(p) for i,p in enumerate(protocol_best_epoch[:,1]) if protocol_best_epoch[i,0]=='Y']
plt.rcParams['font.size'] = '22'
plt.scatter(lX, X, color='red', label='X axis')
plt.scatter(lY, Y, color='blue', label='Y axis')
plt.xlabel('iteration')
plt.ylabel('compression force')
plt.legend()
plt.show()