# DNN-from-scratch
Simple Cat-Non-Cat binary image classifier without using machine learning APIs.

# Notes:
* For the first time you need to train a NN using **DNNTrain.py**.
* You can use **DNNTest** to test an image, copy it to **images** folder and change **my_image** variable name.
* You can change default values of: learning rate, number of iterations, number of layers and number of nodes in each layer.
  ```python
  ### CONSTANTS ###
  in_l = 64*64*3 #W*H*RGBChannels
  layers_dims = [in_l, 20, 7, 5, 1] #  4-layer model
  learning_rate = 0.0075
  num_iterations = 2500
  ```
* The trained NN weights and baises are saved as **parameters.pickle** inside the script directory.
* If **parameters.pickle** doesn't exist the NN will start over. Otherwise it gets initalized with the saved file.
* The used training and test examples are stored using *h5py package*.
* Made by following [Neural Networks and Deep Learning by DeepLearning.AI](https://www.coursera.org/learn/neural-networks-deep-learning).
