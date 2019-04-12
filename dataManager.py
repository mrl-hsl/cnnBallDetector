#data pip line for ball localizor
# mohmmad Hossien Delavaran
import tensorflow as tf
import glob
import os
import cv2
import numpy as np

dataset = tf.data.TFRecordDataset(['./dataSet/dataSet.tfrecords','./fineTune/images.tfrecords'])#,'./fineTune/images.tfrecords'
# test_set = tf.data.TFRecordDataset(['./val/val1.tfrecords','./val/val2.tfrecords','./val/val3.tfrecords'])
test_set = tf.data.TFRecordDataset(['./val/unseenBall.tfrecords'])
def getBox(bbox,windowSize):
    x1 = int(bbox[0]*windowSize[0]) - int(bbox[2]*windowSize[0]/2)
    y1 = int(bbox[1]*windowSize[1]) - int(bbox[3]*windowSize[1]/2)
    w = int(bbox[2]*windowSize[0])
    h = int(bbox[3]*windowSize[1])
    x2 = int(bbox[0]*windowSize[0]) + int(bbox[2]*windowSize[0]/2)
    y2 = int(bbox[1]*windowSize[1]) + int(bbox[3]*windowSize[1]/2)
    return (x1,y1),(x2,y2)

# Create a dictionary describing the features.
def _parse_image_function(example_proto):
  image_feature_description = {
    'height': tf.FixedLenFeature([], tf.int64),
    'width': tf.FixedLenFeature([], tf.int64),
    'depth': tf.FixedLenFeature([], tf.int64),
    'ballX': tf.FixedLenFeature([], tf.float32),
    'ballY': tf.FixedLenFeature([], tf.float32),
    'ballR': tf.FixedLenFeature([], tf.float32),
    'label': tf.FixedLenFeature([], tf.int64),
    'image_raw': tf.FixedLenFeature([], tf.string)
  }
  exm = tf.parse_single_example(example_proto, image_feature_description)
  image_raw = tf.image.decode_jpeg(exm["image_raw"])
  height = tf.cast(exm['height'], tf.float32)
  width = tf.cast(exm['width'], tf.float32)
  
  label = tf.cast(exm['label'], tf.float32)
  ballX = tf.cast(exm['ballX'], tf.float32)/width
  ballY = tf.cast(exm['ballY'], tf.float32)/height
  ballW = 2.0*tf.cast(exm['ballR'], tf.float32)/width
  ballH = 2.0*tf.cast(exm['ballR'], tf.float32)/height
  img =tf.cast(image_raw,dtype=tf.float32)
  oneHot =tf.cast([1.0-label,label],tf.float32) 
  img = tf.image.rgb_to_grayscale(img)

  return img,oneHot,ballX,ballY,ballW,ballH

def color(x,label,ballX,ballY,ballW,ballH):
    # x = tf.image.random_hue(x, 0.5)
    # x = tf.image.random_saturation(x, 0.8, 1.2)
    x = tf.image.random_brightness(x, 0.09)
    # x = tf.image.random_contrast(x, 0.7, 1.7)
    x = tf.clip_by_value(x, 0, 255)
    # x = tf.image.per_image_standardization(x)

    return x,label,ballX,ballY,ballW,ballH
def _resize_data( image,label,ballX,ballY,ballW,ballH):
        """
        Resizes images to specified size.
        """
        image = tf.expand_dims(image, axis=0)
        image = tf.image.resize_images(image,(64,64))
        image = tf.squeeze(image, axis=0)
        # image = tf.image.per_image_standardization(image)
        return image,label,ballX,ballY,ballW,ballH

def filterSmallBall(image,label,ballX,ballY,ballW,ballH):
    return tf.logical_or(tf.greater(ballW,0.25),tf.equal(ballW,0.0))

dataset = dataset.map(_parse_image_function)
dataset = dataset.map(color)

dataset = dataset.map(_resize_data)

dataset = dataset.filter(filterSmallBall)

dataset = dataset.shuffle(buffer_size=12000)

test_set = test_set.map(_parse_image_function)
# test_set = test_set.filter(filterSmallBall)
test_set = test_set.map(_resize_data)
# test_set = test_set.map(color)

test_set = test_set.shuffle(buffer_size=2000)

train_set = dataset


train_set = train_set.batch(32)
test_set = test_set.batch(32)


train_iterator = train_set.make_initializable_iterator()
next_train_batch = train_iterator.get_next()

test_iterator = test_set.make_initializable_iterator()
next_test_batch = test_iterator.get_next()

