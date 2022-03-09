# DNN-from-scratch
Simple Cat-Non-Cat binary image classifier from scratch.

# Notes:
* For the first time you need to train a NN using **DNNTrain.py**.
* You can use **DNNTest.py** to test an image, copy it to **images** folder and change **my_image** variable name.
  ```python
  my_image = "my_image_1.jpg" # change this to the name of your image file in "images\" directory
  my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)
  ```
* You can change default values of learning rate, number of iterations, number of layers and number of nodes in each layer inside **DNNTrain.py**.
  ```python
  ### CONSTANTS ###
  in_l = 64*64*3 #W*H*RGBChannels
  layers_dims = [in_l, 20, 7, 5, 1] #  4-layer model
  learning_rate = 0.0075
  num_iterations = 2500
  ```
* The trained NN weights and biases are saved as **parameters.pickle** inside the script directory.
* If **parameters.pickle** does not exist the NN will start over. Otherwise, it gets initialized with the saved file.
* The used training and test examples are stored using *h5py package*.
* Made by following [Neural Networks and Deep Learning by DeepLearning.AI](https://www.coursera.org/learn/neural-networks-deep-learning).
