'''
Credits: Alberto Castrignanò
            - albertocastrignano2@gmail.com
            - https://www.linkedin.com/in/albertocastrignano2/
            - https://github.com/AlbInitPolito
'''

from subprocess import Popen
from subprocess import DEVNULL
import xml.etree.ElementTree as ET
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from xml.dom import minidom
import numpy as np
import PIL.Image as Image
import time
from multiprocessing import Process
from multiprocessing import Pipe
import math
import env.vtkInterface as vki
import palacellProlifTrain as ppt

base_outfolder = "../../results"
base_palacell_folder = "../../data/PalaCell2D/app"

disc_acts = ["X","Y"]
class PalacellEnv():
  def __init__(self,width=300, height=300, lr=0.001, gamma=0.99, iters=20, max_iterations=4200,
   preload_model_weights=None, preload_losses=None, preload_observations=None, preload_performance=None, preload_data_to_save=None,
   output_dir="palacell_out", mode="prolif", target=[0,0,0], configuration_file='compr.xml', output_file='chem_1-', starting_epoch=0):
    self.configuration_dir = "./"
    self.configuration_file = configuration_file
    self.configuration = self.configuration_dir+configuration_file
    self.iters = iters
    self.epochs = 300
    self.max_iterations=max_iterations
    self.iterations = int(max_iterations/iters)
    self.iteration_num = 0
    self.width = width
    self.height = height
    self.channels = 3
    self.lr = lr
    self.gamma = gamma
    self.preload_model_weights = preload_model_weights
    self.preload_observations = preload_observations
    self.preload_losses = preload_losses
    self.preload_performance = preload_performance
    self.preload_data_to_save = preload_data_to_save
    self.output_dir = output_dir
    if mode=='circle_prolif':
      self.num_continue = 1
      self.num_discrete = 1
      self.range_continue = [(0,10)]
      self.dim_discrete = [2]
    elif mode=='circles':
      self.num_continue = 2
      self.num_discrete = 0
      self.range_continue = [(90,310),(90,310)]
      self.dim_discrete = None
    self.least_iterations = max_iterations
    self.mode = mode
    self.target = target
    self.output_file = output_file
    self.start_time = time.time()
    self.best_cell_num = 0
    self.compr_history = []
    self.epoch_compr = []
    self.best_inside_outside = (0,300)
    self.inside_outside = []
    self.last_inside_outside = (0,0)
    self.performance_updated = False
    self.circle_actions = []
    self.cell_numbers = []
    self.cell_increments = []
    self.epoch_cell_increments = []

    if mode=='circles':
      work1, work2 = Pipe(True)
      prolif_env = PalacellEnv(iters=self.iters, configuration_file='circle_prolif_'+str(self.iters)+'_'+str(self.lr)+'_'+str(self.gamma)+'.xml',
                              output_file='chem_'+str(self.iters)+'_'+str(self.lr)+'_'+str(self.gamma)+'-',
                              output_dir='experiment2/inner/new2_palacell_circle_prolif_out_'+str(self.iters), max_iterations=2500,
                              lr=self.lr, gamma=self.gamma, target=self.target, mode='circle_prolif')
      prolif_env.epochs = self.epochs
      suf = base_outfolder+"/experiment2/inner/new2_palacell_circle_prolif_out_"+str(self.iters)+'_'+str(lr)+'_'+str(gamma)
      if self.preload_model_weights:
        prolif_env.preload_model_weights = suf+"/model_at_epoch_"+str(starting_epoch)+".h5"
      if self.preload_data_to_save:
        prolif_env.preload_data_to_save = suf+"/data_to_save_at_epoch_"+str(starting_epoch)
      if self.preload_performance:
        prolif_env.preload_performance = suf+"/performance_at_epoch_"+str(starting_epoch)
      self.prolif_env = prolif_env
      prolif_train = ppt.ProlifTrain(self.prolif_env, self.lr, self.gamma)
      prolif_proc = Process(target=prolif_train.train, args=[5, False, starting_epoch, work1])
      prolif_proc.start()
      self.prolif_train = prolif_train
      self.prolif_proc = prolif_proc
      self.pipe = work2

  def reset(self): #todo!
    cwd = os.getcwd()
    os.chdir(base_palacell_folder)
    if self.mode == 'circles':
      observation = np.asarray(Image.new("RGB", (self.width,self.height), (0,0,0))).copy()
      observation = vki.add_target(observation, [self.target[0]*3/4, self.target[1]*3/4], self.target[2]*3/4).reshape((1,self.width,self.height,3)).copy()
    elif self.mode == 'circle_prolif':
      self.last_cell_num = 0
      self.epoch_compr = []
      self.epoch_cell_increments = []
      self.iteration_num = 0
      try:
        process = Popen(['./palaCell',self.configuration], stdout=DEVNULL)
        process.wait()
      except Exception as e:
        os.chdir(cwd)
        raise Exception(e)
      image = vki.create_pil_image("output/"+self.output_file+"_final_cell")
      observation = vki.pil_to_array(image).reshape((1,self.width,self.height,3))
    os.chdir(cwd)
    return observation

  def _get_info(self):
    st = ""
    if self.mode == 'circle_prolif':
      st = "last cell num: "+str(self.last_cell_num)+", best least iterations num: "+str(self.least_iterations)+", best cell num: "+str(self.best_cell_num)
    elif self.mode=='circles':
      st = "last inside-outside: "+str(self.last_inside_outside)+", best last inside-outside: "+str(self.best_inside_outside)
    st = st+" elapsed time: "+str(time.time()-self.start_time)+" iteration: "+str(self.iteration_num)+" iters: "+str(self.iters)
    return st

  def load_data_to_save(self, data):
    if self.mode == 'circle_prolif':
      self.cell_numbers = data['cell_numbers']
      self.cell_increments = data['cell_increments']
      self.compr_history = data['compr_history']
    elif self.mode=='circles':
      self.inside_outside = data['inside_outside']
      self.circle_actions = data['circle_actions']

  def data_to_save(self):
    out = {}
    if self.mode == 'circle_prolif':
      out['cell_numbers'] = self.cell_numbers
      out['cell_increments'] = self.cell_increments
      out['compr_history'] = self.compr_history
    elif self.mode == 'circles':
      out['inside_outside'] = self.inside_outside
      out['circle_actions'] = self.circle_actions
    return out

  def step(self, action):
    cwd = os.getcwd()
    os.chdir(base_palacell_folder)

    if self.mode == 'circle_prolif':
      self.epoch_compr.append(action)
      cont_action = float(action[1])
      cont_action = math.floor(cont_action*(10)+0.5)/(10**3)
      action[1] = str(cont_action)
      self.configure(self.configuration, self.iters, action[0], action[1], export_step=self.iters, initialPath = "output/"+self.output_file+"_final_cell.vtp",
        finalPath = self.output_file)
      try:
        process = Popen(['./palaCell',self.configuration], stdout=DEVNULL)
        process.wait()
      except Exception as e:
        os.chdir(cwd)
        raise Exception(e)

      os.chdir(cwd)
      observation = img_to_array(self.render()).reshape((1,self.width,self.height,3))
      cwd = os.getcwd()
      os.chdir(base_palacell_folder)
      cell_num = vki.read_cell_num("output/"+self.output_file+"_final_cell")
      if cell_num>self.best_cell_num:
        self.best_cell_num = cell_num
        self.performance_updated = True
      reward = cell_num - self.last_cell_num
      self.epoch_cell_increments.append(reward)
      self.last_cell_num = cell_num
      self.iteration_num += self.iters

      if cell_num>=300 or self.iteration_num>=self.max_iterations:
        done = True
      else:
        done = False
      if done:
        self.compr_history.append(self.epoch_compr)
        print()
        print(self.epoch_compr)
        print()
        if self.iteration_num < self.least_iterations:
          self.least_iterations = self.iteration_num 
        self.cell_numbers.append(cell_num)
        self.cell_increments.append(self.epoch_cell_increments)
    elif self.mode == 'circles':
      print(action)
      self.circle_actions.append(action)
      self.configure(self.configuration,0,finalPath = 'chem_'+str(self.iters)+'_'+str(self.lr)+'_'+str(self.gamma)+'-',initial_position=np.array(action))
      
      '''
      wait for training process of prolif
      '''
      self.pipe.send(self.configuration)
      out_path = self.pipe.recv()

      done = True
      outside, inside = vki.count_target_points(out_path, [self.target[0], self.target[1]], self.target[2])
      cell_num = vki.read_cell_num(out_path)
      reward = (inside-outside)/cell_num
      self.inside_outside.append((inside/cell_num,outside/cell_num))
      diff = self.best_inside_outside[0]-self.best_inside_outside[1]
      if diff < (inside-outside)/cell_num or (diff == (inside-outside)/cell_num and self.best_inside_outside[0]<inside/cell_num):
        self.best_inside_outside = (inside/cell_num, outside/cell_num)
        self.performance_updated = True
      self.last_inside_outside = (inside/cell_num, outside/cell_num)
      print("inside/outside: "+str(inside/cell_num)+"/"+str(outside/cell_num))
      image = vki.create_decentered_pil_image(out_path).resize([self.width, self.height])
      observation = vki.pil_to_array(image)
      observation = vki.add_target(observation, [self.target[0]*3/4, self.target[1]*3/4], self.target[2]*3/4).reshape((1,self.width,self.height,3)).copy()
    os.chdir(cwd)
    return observation, reward, done, None

  def render(self):
    cwd = os.getcwd()
    os.chdir(base_palacell_folder)
    if self.mode == 'circle_prolif':
      image = vki.create_pil_image("output/"+self.output_file+"_final_cell").resize([self.width,self.height])
    elif self.mode=='circles':
      observation = np.asarray(Image.new("RGB", (self.width,self.height), (0,0,0))).copy()
      observation = vki.add_target(observation, [self.target[0]*3/4, self.target[1]*3/4], self.target[2]*3/4).reshape((1,self.width,self.height,3)).copy()
      image = vki.array_to_pil(observation)
    os.chdir(cwd)
    return image

  def adapt_actions(self, discrete_actions = None, continue_actions=None):
    if self.mode == 'circle_prolif':
      return [disc_acts[discrete_actions[0]],str(continue_actions[0].numpy())]
    elif self.mode=='circles':
      return [continue_actions[0], continue_actions[1]]

  def save_performance(self, values):
    if self.mode == 'circle_prolif':
      self.performance_indexes = self.best_cell_num
    elif self.mode == 'circles':
      self.performance_indexes = self.best_inside_outside
    return

  def load_performance(self, values):
    if self.mode == 'circle_prolif':
      self.best_cell_num = values
      self.performance_indexes = self.best_cell_num
    elif self.mode == 'circles':
      self.best_inside_outside = values
      self.performance_indexes = self.best_inside_outside
    return

  def check_performance(self, values):
    if self.performance_updated:
      self.performance_updated = False
      return True
    if self.mode=='circle_prolif':
      if self.best_cell_num >= self.last_cell_num:
        return False
      return True
    elif self.mode == 'circles':# or self.mode == 'circle_prolif':
      diff = self.best_inside_outside[0] - self.best_inside_outside[1]
      diff2 = self.last_inside_outside[0] - self.last_inside_outside[1]
      if diff < diff2 or (diff==diff2 and self.last_inside_outside[0]>self.best_inside_outside[0]):
        return True
      return False

  def get_performance(self):
    return self.performance_indexes

  def configure(self, filePath, num_iter=5, axis='X', compr_force=0.0, export_step='0', init='0', initialPath = ' ', initialWallPath = ' ', initial_position = [200,200], finalPath = 'chem_1-'):
    root = ET.Element('parameters')

    geometry = ET.SubElement(root, 'geometry')
    simulation = ET.SubElement(root, 'simulation')
    physics = ET.SubElement(root, 'physics')
    numerics = ET.SubElement(root, 'numerics')

    #geometry
    initialVTK = ET.SubElement(geometry, 'initialVTK')
    initialWallVTK = ET.SubElement(geometry, 'initialWallVTK')
    finalVTK = ET.SubElement(geometry, 'finalVTK')
    finalVTK.text = finalPath
    initialVTK.text = initialPath
    initialWallVTK.text = initialWallPath

    #simulation
    type = ET.SubElement(simulation,"type")
    exportStep = ET.SubElement(simulation,"exportStep")
    initStep = ET.SubElement(simulation,"initStep")
    verbose = ET.SubElement(simulation,"verbose")
    exportCells = ET.SubElement(simulation,"exportCells")
    exportForces = ET.SubElement(simulation,"exportForces")
    exportField = ET.SubElement(simulation,"exportField")
    exportSpecies = ET.SubElement(simulation,"exportSpecies")
    exportDBG = ET.SubElement(simulation,"exportDBG")
    exportCSV = ET.SubElement(simulation,"exportCSV")
    seed = ET.SubElement(simulation,"seed")
    exit = ET.SubElement(simulation,"exit")
    numIter = ET.SubElement(simulation,"numIter")
    numTime = ET.SubElement(simulation,"numTime")
    numCell = ET.SubElement(simulation,"numCell")
    startAt = ET.SubElement(simulation,"startAt")
    stopAt = ET.SubElement(simulation,"stopAt")
    initialPos = ET.SubElement(simulation,"initialPos")

    type.text = '1'
    exportStep.text = str(export_step)
    initStep.text = init
    verbose.text = '0'
    exportCells.text = "true"
    exportForces.text = "false"
    exportField.text = "false"
    exportSpecies.text = "false"
    exportDBG.text = "false"
    exportCSV.text = "false"
    seed.text = '40' #-1?
    exit.text = "iter"
    numIter.text = str(num_iter)
    numTime.text = '7200'
    numCell.text = '300'
    startAt.text = '0'
    stopAt.text = str(num_iter)
    initialPos.text = str(initial_position[0])+" "+str(initial_position[1])

    #physics
    diffusivity = ET.SubElement(physics, "diffusivity")
    reactingCell = ET.SubElement(physics, "reactingCell")
    reactionRate = ET.SubElement(physics, "reactionRate")
    dissipationRate = ET.SubElement(physics, "dissipationRate")
    growthThreshold = ET.SubElement(physics, "growthThreshold")
    zeta = ET.SubElement(physics, "zeta")
    rho0 = ET.SubElement(physics, "rho0")
    d0 = ET.SubElement(physics, "d0")
    dmax = ET.SubElement(physics, "dmax")
    n0 = ET.SubElement(physics, "n0")
    numCells = ET.SubElement(physics, "numCells")
    cell = ET.SubElement(physics, "cell")
    numVertex = ET.SubElement(physics, "numVertex")
    edgeVertex = ET.SubElement(physics, "edgeVertex")
    vertex = ET.SubElement(physics, "vertex")
    vertex_1 = ET.SubElement(physics, "vertex")
    vertex_2 = ET.SubElement(physics, "vertex")
    extern = ET.SubElement(physics, "extern")

    diffusivity.text = '2.0'
    reactingCell.text = '0'
    reactionRate.text = '0.001'
    dissipationRate.text = '0.0001'
    growthThreshold.text = '0.025'
    zeta.text = '0.7'
    rho0.text = '1.05'
    d0.text = '0.5'
    dmax.text = '1.0'
    n0.text = '123'
    numCells.text = '1'
    cell.attrib['type'] = 'default'
    numVertex.text = '3'
    edgeVertex.text = '-1'
    vertex.attrib['type'] = 'default'
    vertex_1.attrib['type'] = '1'
    vertex_2.attrib['type'] = '2'

    #physics.cell
    divisionThreshold = ET.SubElement(cell,"divisionThreshold")
    pressureSensitivity = ET.SubElement(cell,"pressureSensitivity")
    nu = ET.SubElement(cell,"nu")
    nuRelax = ET.SubElement(cell,"nuRelax")
    A0 = ET.SubElement(cell,"A0")
    k4 = ET.SubElement(cell,"k4")
    probToProlif = ET.SubElement(cell,"probToProlif")
    maxPressureLevel = ET.SubElement(cell,"maxPressureLevel")
    zetaCC = ET.SubElement(cell, "zetaCC")

    divisionThreshold.text = '300.0'
    pressureSensitivity.text = '2.5'
    nu.text = '0.0025'
    nuRelax.text = '0.01'
    A0.text = '300.0'
    k4.text = '0.01'
    probToProlif.text = '0.001'
    maxPressureLevel.text = '0.05'
    zetaCC.text = '0.4'

    #physics.vertex
    k1 = ET.SubElement(vertex, "k1")
    k3 = ET.SubElement(vertex, "k3")
    k1.text = '0.2'
    k3.text = '0.2'
    k1_1 = ET.SubElement(vertex_1, "k1")
    k1_2 = ET.SubElement(vertex_2, "k3")
    k1_1.text = '0.001'
    k1_2.text = '0.2'

    #physics.extern
    compressionAxis = ET.SubElement(extern, "compressionAxis")
    comprForce = ET.SubElement(extern, "comprForce")
    kExtern = ET.SubElement(extern, "kExtern")
    center = ET.SubElement(extern, "center")
    rMin = ET.SubElement(extern, "rMin")

    compressionAxis.text = axis
    comprForce.text = str(compr_force)
    kExtern.text = '0.01'
    center.text = '200 200'
    rMin.text = '100'

    #numerics
    dx = ET.SubElement(numerics, "dx")
    dt = ET.SubElement(numerics, "dt")
    domain = ET.SubElement(numerics, "domain")
    
    dx.text = '1.0'
    dt.text = '1.0'
    domain.text = '0 0 400. 400.'

    tree = ET.ElementTree(root)
    #ET.indent(root)
    #tree.write(filePath, xml_declaration=True)
    indented_xml = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    with open(filePath, "w") as f:
        f.write(indented_xml)