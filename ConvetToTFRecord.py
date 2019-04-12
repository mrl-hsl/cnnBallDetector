# converting image data to TFRecords.
# mhdelavaran.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import tensorflow as tf
tf.enable_eager_execution()
import glob
import numpy as np
import IPython.display as display
import cv2

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
  
def image_example(image_string, ballX,ballY,ballR,label):
  image_shape = tf.image.decode_jpeg(image_string).shape

  feature = {
      'height': _int64_feature(image_shape[0]),
      'width': _int64_feature(image_shape[1]),
      'depth': _int64_feature(image_shape[2]),
      'ballX': _float_feature(ballX),
      'ballY': _float_feature(ballY),
      'ballR': _float_feature(ballR),
      'label': _int64_feature(label),
      'image_raw': _bytes_feature(image_string),
  }
  return tf.train.Example(features=tf.train.Features(feature=feature))

with tf.python_io.TFRecordWriter('./val/unseenBall.tfrecords') as writer:
    with open('./val/UnSeenBall/bbBoxs.txt') as f:
        lines = f.readlines()
    counter = 0
    for line in lines:
        st = line.split(' ')
        fileName =st[0]
        if os.path.isfile("./val/UnSeenBall/Temp/"+fileName):
            print(counter)
            image = cv2.imread("./val/UnSeenBall/Temp/"+fileName)
            # image = cv2.resize(image, (32, 32)) 
            ballX = float(st[1])
            ballY = float(st[2])
            ballR = float(st[3])
            counter = counter + 1
            label = 1
            image_string = cv2.imencode('.jpg', image)[1].tostring()
            tf_example = image_example(image_string, ballX,ballY,ballR,label)
            writer.write(tf_example.SerializeToString())
    # for fileAddress in glob.glob('./val/neg/*'):
    #     print(counter)
    #     image = cv2.imread(fileAddress)
    #     counter = counter + 1
    #     # if counter >4000:
    #     #   break
    #     ballX = 0.0
    #     ballY = 0.0
    #     ballR = 0.0
    #     label = 0
    #     image_string = cv2.imencode('.jpg', image)[1].tostring()
    #     tf_example = image_example(image_string, ballX,ballY,ballR,label)
    #     writer.write(tf_example.SerializeToString())
