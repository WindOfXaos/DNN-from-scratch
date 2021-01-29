import os,sys,inspect
import pickle
import numpy as np
import matplotlib.pyplot as plt
import imageio
from skimage import transform
from utils import *
from DNNTrain import num_px,classes

#detection machine
from sys import platform as _platform
if _platform == "win32":
    scriptPATH = os.path.abspath(inspect.getsourcefile(lambda:0)) # compatible interactive Python Shell
    scriptDIR  = os.path.dirname(scriptPATH)

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
np.random.seed(1)

#Loads previous parameters if exist
parameters = None
for filename in os.listdir(scriptDIR):
    if filename.endswith(".pickle"):
        with open('parameters.pickle', 'rb') as f:
            parameters = pickle.load(f)

if not parameters:
    print("Cannot find parameters.pickle, train a network using DNNTrain.py to proceed.")
    sys.exit()
    
    

my_image = "my_image_1.jpg" # change this to the name of your image file in "images\" directory
my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)

fname = os.path.join(scriptDIR,"images", my_image)
try:
    image = np.array(imageio.imread(fname, format=None))
except IOError:
    print ("No such file exists >>", my_image)
    sys.exit()
my_image = transform.resize(image, (num_px,num_px)).reshape((num_px*num_px*3,1))
my_predicted_image = predict(my_image, my_label_y, parameters)

print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")

plt.imshow(image)
plt.show()