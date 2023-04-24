'''
Credits: Alberto Castrignanò
            - albertocastrignano2@gmail.com
            - https://www.linkedin.com/in/albertocastrignano2/
            - https://github.com/AlbInitPolito
'''

import numpy as np
import matplotlib.pyplot as plt

# for epochs 0-70:

####################################################################################
################################# LOAD OUTPUT DATA #################################
####################################################################################

data_001_095 = np.load("../../results/experiment2/new_palacell_out_circles_0.001_0.95/data_to_save_at_epoch_70.npy", allow_pickle=True).item()
data_0001_095 = np.load("../../results/experiment2/new_palacell_out_circles_0.0001_0.95/data_to_save_at_epoch_70.npy", allow_pickle=True).item()
data_001_099 = np.load("../../results/experiment2/new_palacell_out_circles_0.001_0.99/data_to_save_at_epoch_70.npy", allow_pickle=True).item()
data_0001_099 = np.load("../../results/experiment2/new_palacell_out_circles_0.0001_0.99/data_to_save_at_epoch_70.npy", allow_pickle=True).item()
data = [data_001_095, data_001_099, data_0001_095, data_0001_099]

####################################################################################
################################ OBTAIN BEST RESULT ################################
####################################################################################

print("lr: 0.001 gamma: 0.95 max cells: ", max(np.array(data_001_095['inside_outside'])[:, 0]))
print("lr: 0.0001 gamma: 0.95 max cells: ", max(np.array(data_0001_095['inside_outside'])[:, 0]))
print("lr: 0.001 gamma: 0.99 max cells: ", max(np.array(data_001_099['inside_outside'])[:, 0]))
print("lr: 0.0001 gamma: 0.99 max cells: ", max(np.array(data_0001_099['inside_outside'])[:, 0]))

# find best epoch
best_configuration = np.array(data_0001_099['inside_outside'])[:,0]
best_epoch = np.where(best_configuration==max(best_configuration)) # should be 37

####################################################################################
################################# GENERATE FIGURES #################################
####################################################################################

x = list(range(71)) # x-axis

# plot the final number of cells for each hyperparameters configuration
plt.rcParams['font.size'] = '22'
plt.xlabel('epoch')
plt.ylabel('fraction of cells inside the target wrt total amount')
plt.plot(x, np.array(data_001_095['inside_outside'])[:,0], label="lr=0.001 gamma=0.95")
plt.plot(x, np.array(data_001_099['inside_outside'])[:,0], label="lr=0.001 gamma=0.99")
plt.plot(x, np.array(data_0001_095['inside_outside'])[:,0], label="lr=0.0001 gamma=0.95")
plt.plot(x, np.array(data_0001_099['inside_outside'])[:,0], label="lr=0.0001 gamma=0.99")
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
    for j in range(4):
        window = np.array(data[j]['inside_outside'][10*i:10*i+20])
        means[j].append(np.mean(window))
        vars[j].append(np.var(window))

# plot mean of the 6 windows
x = list(range(6))
plt.rcParams['font.size'] = '22'
plt.plot(x, means[0], label="lr=0.001 gamma=0.95")
plt.plot(x, means[1], label="lr=0.001 gamma=0.99")
plt.plot(x, means[2], label="lr=0.0001 gamma=0.95")
plt.plot(x, means[3], label="lr=0.0001 gamma=0.99")
plt.xlabel('window')
plt.ylabel('mean of fraction of cells inside the target wrt total amount')
plt.legend()
plt.show()

# plot var of the 6 windows
plt.rcParams['font.size'] = '22'
plt.plot(x, vars[0], label="lr=0.001 gamma=0.95")
plt.plot(x, vars[1], label="lr=0.001 gamma=0.99")
plt.plot(x, vars[2], label="lr=0.0001 gamma=0.95")
plt.plot(x, vars[3], label="lr=0.0001 gamma=0.99")
plt.xlabel('window')
plt.ylabel('variance of fraction of cells inside the target wrt total amount')
plt.legend()
plt.show()

# plot relative starting position coordinates
x = list(range(71))
plt.rcParams['font.size'] = '22'
plt.xlabel('iteration')
plt.ylabel('value of relative starting position coordinate')
plt.plot(x, np.array(data_001_099['circle_actions'])[:,0], label="X coordinate")
plt.plot(x, np.array(data_001_099['circle_actions'])[:,1], label="Y coordinate")
plt.legend()
plt.show()

# retrieve values of biofabrication protocol for epoch 0 and for best epoch (should be epoch 37)
#print(best_epoch)
compr_history = np.load("../../results/experiment2/inner/new2_palacell_circle_prolif_out_20_0.0001_0.99/data_to_save_at_epoch_70.npy", allow_pickle=True).item()
protocol_epoch_0 = np.array(compr_history['compr_history'])[0, 0:125]
protocol_best_epoch = np.array(compr_history['compr_history'])[37, 37*125:38*125]

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


#for epochs 70-140:

####################################################################################
################################# LOAD OUTPUT DATA #################################
####################################################################################

data_001_095 = np.load("../../results/experiment2/new2_palacell_out_0.001_0.95/data_to_save_at_epoch_140.npy", allow_pickle=True).item()
data_0001_095 = np.load("../../results/experiment2/new2_palacell_out_0.0001_0.95/data_to_save_at_epoch_140.npy", allow_pickle=True).item()
data_001_099 = np.load("../../results/experiment2/new2_palacell_out_0.001_0.99/data_to_save_at_epoch_140.npy", allow_pickle=True).item()
data_0001_099 = np.load("../../results/experiment2/new2_palacell_out_0.0001_0.99/data_to_save_at_epoch_140.npy", allow_pickle=True).item()
data = [data_001_095, data_001_099, data_0001_095, data_0001_099]

####################################################################################
################################ OBTAIN BEST RESULT ################################
####################################################################################

print("lr: 0.001 gamma: 0.95 max cells: ", max(np.array(data_001_095['inside_outside'])[:, 0]))
print("lr: 0.0001 gamma: 0.95 max cells: ", max(np.array(data_0001_095['inside_outside'])[:, 0]))
print("lr: 0.001 gamma: 0.99 max cells: ", max(np.array(data_001_099['inside_outside'])[:, 0]))
print("lr: 0.0001 gamma: 0.99 max cells: ", max(np.array(data_0001_099['inside_outside'])[:, 0]))

# find best epoch
best_configuration = np.array(data_0001_099['inside_outside'])[:,0]
best_epoch = np.where(best_configuration==max(best_configuration)) # should be 135

# from now on, work only on epochs 70-140
data_001_095['inside_outside'] = data_001_095['inside_outside'][70:141]
data_001_095['circle_actions'] = data_001_095['circle_actions'][70:141]
data_001_099['inside_outside'] = data_001_099['inside_outside'][70:141]
data_001_099['circle_actions'] = data_001_099['circle_actions'][70:141]
data_0001_095['inside_outside'] = data_0001_095['inside_outside'][70:141]
data_0001_095['circle_actions'] = data_0001_095['circle_actions'][70:141]
data_0001_099['inside_outside'] = data_0001_099['inside_outside'][70:141]
data_0001_099['circle_actions'] = data_0001_099['circle_actions'][70:141]

####################################################################################
################################# GENERATE FIGURES #################################
####################################################################################

x = list(range(71)) # x-axis

# plot the final number of cells for each hyperparameters configuration
plt.rcParams['font.size'] = '22'
plt.xlabel('epoch')
plt.ylabel('fraction of cells inside the target wrt total amount')
plt.plot(x, np.array(data_001_095['inside_outside'])[:,0], label="lr=0.001 gamma=0.95")
plt.plot(x, np.array(data_001_099['inside_outside'])[:,0], label="lr=0.001 gamma=0.99")
plt.plot(x, np.array(data_0001_095['inside_outside'])[:,0], label="lr=0.0001 gamma=0.95")
plt.plot(x, np.array(data_0001_099['inside_outside'])[:,0], label="lr=0.0001 gamma=0.99")
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
    for j in range(4):
        window = np.array(data[j]['inside_outside'][10*i:10*i+20])
        means[j].append(np.mean(window))
        vars[j].append(np.var(window))

# plot mean of the 6 windows
x = list(range(6))
plt.rcParams['font.size'] = '22'
plt.plot(x, means[0], label="lr=0.001 gamma=0.95")
plt.plot(x, means[1], label="lr=0.001 gamma=0.99")
plt.plot(x, means[2], label="lr=0.0001 gamma=0.95")
plt.plot(x, means[3], label="lr=0.0001 gamma=0.99")
plt.xlabel('window')
plt.ylabel('mean of fraction of cells inside the target wrt total amount')
plt.legend()
plt.show()

# plot var of the 6 windows
plt.rcParams['font.size'] = '22'
plt.plot(x, vars[0], label="lr=0.001 gamma=0.95")
plt.plot(x, vars[1], label="lr=0.001 gamma=0.99")
plt.plot(x, vars[2], label="lr=0.0001 gamma=0.95")
plt.plot(x, vars[3], label="lr=0.0001 gamma=0.99")
plt.xlabel('window')
plt.ylabel('variance of fraction of cells inside the target wrt total amount')
plt.legend()
plt.show()

# plot relative starting position coordinates
x = list(range(71))
plt.rcParams['font.size'] = '22'
plt.xlabel('iteration')
plt.ylabel('value of relative starting position coordinate')
plt.plot(x, np.array(data_001_099['circle_actions'])[:,0], label="X coordinate")
plt.plot(x, np.array(data_001_099['circle_actions'])[:,1], label="Y coordinate")
plt.legend()
plt.show()

# retrieve values of biofabrication protocol for epoch 0 and for best epoch (should be epoch 135)
#print(best_epoch)
compr_history = np.load("../../results/experiment2/inner/new2_palacell_circle_prolif_out_20_0.0001_0.99/data_to_save_at_epoch_140.npy", allow_pickle=True).item()
protocol_epoch_0 = np.array(compr_history['compr_history'])[0, 0:125]
protocol_best_epoch = np.array(compr_history['compr_history'])[135, 135*125:136*125]

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