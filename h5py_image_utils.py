import h5py
import numpy as np
import matplotlib.pyplot as plt

# function to load data.
def load_h5py_data(index, show_example=True):
    train_dataset = h5py.File('/content/drive/MyDrive/Colab Notebooks/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('/content/drive/MyDrive/Colab Notebooks/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    if show_example:
        plt.imshow(train_set_x_orig[index])
        print ("y = " + str(train_set_y_orig[0,index]) + ". It's a " + classes[train_set_y_orig[0,index]].decode("utf-8") +  " picture.")

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig


# function to see the shapes of the data.
def see_shapes(X_train, X_test, Y_train, Y_test):

  m_train = X_train.shape[0]
  m_test = X_test.shape[0]

  if len(X_train.shape) == 3:
    image_size = X_train.shape[1]

  print ("Number of training examples: m_train = ", m_train)
  print ("Number of testing examples: m_test = ", m_test)

  if len(X_train.shape) == 3:
    print ("Height/Width of each image: num_px = ", image_size)
    print ("Each image is of size: ", image_size, "x", image_size)
    
  print ("X_train shape: ", X_train.shape)
  print ("Y_train shape: ", Y_train.shape)
  print ("X_test shape: ", X_test.shape)
  print ("Y_test shape: ", Y_test.shape)


# Flatten the images.
def flatten(X_train, X_test, Y_train, Y_test):

  ### BEGIN CODE HERE ###

  X_train = X_train.reshape(X_train.shape[0], -1).T
  X_test = X_test.reshape(X_test.shape[0], -1).T

  ### END CODE HERE ###

  print ("Flattened X_train shape: " + str(X_train.shape))
  print ("Y_train shape: " + str(Y_train.shape))
  print ("Flattened X_test shape: " + str(X_test.shape))
  print ("Y_test shape: " + str(Y_test.shape))

  return X_train, X_test


# Basic normalization.
def normalize(X_train, X_test):

  ### BEGIN CODE HERE

  X_train = X_train/255.
  X_test = X_test/255.

  ### END CODE HERE
  
  return X_train, X_test