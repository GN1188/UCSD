import tensorflow as tf
import tensorflow_datasets as tfds 
import numpy as np

def load_data():
  (ds_train, ds_test), ds_info = tfds.load('cats_vs_dogs',
      split=['train[:10%]', 'train[:2%]'], with_info=True, as_supervised=True)

  return (ds_train, ds_test), ds_info

### END CODE HERE

# Preprocess the data using your helper functions.
def preprocess(ds, ds_info, batched=False):
  """
  Resizes images, one hot encodes labels, shuffles and batches data.
  Arguments: 
  ds: Your TensorFlow dataset.
  Returns:
  ds: Your preprocessed TensorFlow dataset.
  """
  # Use the dataset info to extract the number of classes.
  NUM_CLASSES = ds_info.features['label'].num_classes

  ### END CODE HERE

  # If only two classes, set NUM_CLASSES to 1. This will make binary classification easier since we will be using one hot vectors.
  if NUM_CLASSES == 2:
    NUM_CLASSES = 1

  ### BEGIN CODE HERE

  # Use the dataset info to extract the number of channels in the image.
  image_dims = ds_info.features['image'].shape[-1]

  if batched == True:
    BATCH_SIZE = 32

  # Transform labels to categorical vectors with one hot encoding.
  def one_hot(image, label):
    """
    Converts the label to categorical.
    Arguments ~
    image: Tensor of Shape (IMAGE_SIZE,IMAGE_SIZE,image_dims) - Simply for outputting
    label: Tensor of Shape (BATCH_SIZE,) for casting and converting to categorical
    Returns the image (as it was inputted) and the label converted to a categorical vector.
    """
    ### BEGIN CODE HERE

    # Cast to int32.
    label = tf.cast(label, tf.int32)

    # Encode the label as a one hot vector.
    label = tf.one_hot(label, NUM_CLASSES)

    # Recast the label to float32.
    label = tf.cast(label, tf.float32)

    ### END CODE HERE 
    
    return image, label

  ### BEGIN CODE HERE

  # Choose the size you want to resize your image to.
  IMAGE_SIZE = 224

  ### END CODE HERE

  # Resize the images to the desired shape.
  def resize(image, label):
    """
    Resizes the image to (IMAGE_SIZE,IMAGE_SIZE,image_dims) size
    Arguments:
        x: Tensor of Shape (None, None, image_dims) ~ The tensor to be resized
        y: Tensor of Shape (1,) ~ The ground truth label (not transformed, but required for inputting)
    Returns: A tuple of the Resized Image and Label
    """
    ### BEGIN CODE HERE

    # Resize the image using tensorflow.
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])

    ### END CODE HERE

    return (image,label)

### BEGIN CODE HERE

  # Resize all images.
  ds = ds.map(resize)

  # Encode the labels as one hot vectors.
  ds = ds.map(one_hot)

  # Normalize image pixel values using tensorflow.
  ds = ds.map(lambda x, y: (tf.image.per_image_standardization(x), y))

  # Shuffle the dataset.
  ds = ds.shuffle(len(ds))

  if batched == True:
    # Batch the dataset.
    ds = ds.batch(BATCH_SIZE)

  ### END CODE HERE

  print("Preprocessing complete.")

  if batched == True:
    # Check that resizing and batching were executed correctly.
    examples = ds.take(2)
    for image, label in examples:
        print("The batch contains {} examples each of shape {}.".format(image.shape[0], image.shape[1:]))
        print ("After one hot encoding, the labels are of shape {}.".format(label.shape[-1]))

  # Check that resizing and batching were executed correctly.
  examples = ds.take(2)
  for image, label in examples:
    print("The images are now of shape {}.".format(image.shape))
    print ("After one hot encoding, the labels are of shape {}.".format(label.shape))

  return ds

def as_numpy(ds):
  ds_numpy = tfds.as_numpy(ds)
  X = np.array(list(map(lambda x: x[0], ds_numpy)))
  Y = np.array(list(map(lambda x: x[1], ds_numpy)))
  return X, Y