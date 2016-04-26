
# coding: utf-8

# In[ ]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from PIL import Image
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

IMAGE_SIZE_X = 600
IMAGE_SIZE_Y = 450

NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 998
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 200


def read_cifar10(path):
  
  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()

  images = []
  labels = []
  label_bytes = 1  # 2 for CIFAR-100
  result.height = 600
  result.width = 800
  result.depth = 3
  #IMAGE_PIXELS = 600*800*3
  #batch_size = 128
  path_train = "/project/cifar10/images/train/"
  path_test = "/project/cifar10/images/test/"
  if path == "train":
    path_new = path_train
  else:
    path_new = path_test
    
  for root, dirs, files in os.walk(path_new):
    
    for f in files:
        try:
            path_i = os.path.join(root, f)
            if root.split('/')[-1] == 'expressionism':
                image = Image.open(path_i)
                array_3 = np.array(image)
                array_1 = np.reshape(array_3,result.width*result.height*result.depth)
                images.append(array_1)
                labels.append(0)           
            elif root.split('/')[-1] == 'impressionism':
                image = Image.open(path_i)
                array_3 = np.array(image)
                array_1 = np.reshape(array_3,result.width*result.height*result.depth)
                images.append(array_1)
                labels.append(1) 
            elif root.split('/')[-1] == 'postImpressionism':
                image = Image.open(path_i)
                array_3 = np.array(image)
                array_1 = np.reshape(array_3,result.width*result.height*result.depth)
                images.append(array_1)
                labels.append(2)
        except:
            pass
  dic = {'labels':labels, 'data':images}


  labels = tf.placeholder(tf.int32, shape=[None, 1])
  images = tf.placeholder(tf.int32, shape=[None, result.height*result.width*result.depth])
  feed_dict = {labels:dic['labels'], images:dic['data']}
  result.uint8image = tf.reshape(images, [result.height, result.width, result.depth])
  result.label = labels
  return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.
  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.
  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 5
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.image_summary('images', images)

  return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(batch_size):
  path = "train"
  read_input = read_cifar10(path)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE_Y
  width = IMAGE_SIZE_X

  distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

  # Randomly flip the image horizontally.
  distorted_image = tf.image.random_flip_left_right(distorted_image)

  # Because these operations are not commutative, consider randomizing
  # the order their operation.
  distorted_image = tf.image.random_brightness(distorted_image,
                                               max_delta=63)
  distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_whitening(distorted_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)


def inputs(batch_size):
  path = "test"
  read_input = read_cifar10(path)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE_Y
  width = IMAGE_SIZE_X

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         width, height)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_whitening(resized_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)

