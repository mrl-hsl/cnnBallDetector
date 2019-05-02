import tensorflow as tf
import glob
import os
import cv2
class network():
    def __init__(self):
        self.input = tf.placeholder(tf.float32, [None, 64, 64,1], name='input')
        self.isTraning = tf.placeholder(tf.bool, [], name='isTraning')
        self.nboxesInFeatures = [2, 3]
        self.sizeOfFeatures = [2, 1]
        self.scalesInFeatures = [[1/3.0, 1/4.0], [0.33, 0.5, 0.75]]
        # self.nboxesInFeatures = [3]
        # self.sizeOfFeatures = [1]
        # self.scalesInFeatures = [[0.33, 0.5, 0.75]]

        self.activation = tf.nn.leaky_relu
        self.weightInitializer = tf.contrib.layers.xavier_initializer
        # self.nboxesInFeatures = [1]
        # self.sizeOfFeatures = [1]
        # self.scalesInFeatures = [[1.0/2]]
        self.featureLayers = []
        self.normalizer_fn = tf.contrib.layers.batch_norm
        self.norm_params = {"is_training": self.isTraning}
        self.regularizer_fn = None#tf.contrib.layers.l2_regularizer(0.01)

    def conv2d(self,input,nFilters,kernelSize,_strides,_name):
        output = tf.contrib.layers.conv2d(
            inputs= input,
            num_outputs= nFilters,
            kernel_size= kernelSize,
            stride= _strides,
            scope= _name,
            weights_initializer= self.weightInitializer(),
            normalizer_fn= self.normalizer_fn ,
            normalizer_params= self.norm_params,
            activation_fn= self.activation,
            weights_regularizer= self.regularizer_fn,
            padding="SAME"
        )
        return output
    def maxPooling(self,inputs, kernelSize, strides):
        return tf.contrib.layers.max_pool2d(
                    inputs,
                    kernelSize,
                    stride=strides,
                    padding='VALID'
                )

    def avgPooling(self,inputs, kernelSize, strides):
        return tf.contrib.layers.avg_pool2d(
                    inputs,
                    kernelSize,
                    stride=strides,
                    padding='VALID'
                )

    def sepConvMobileNet(self,features,kernel_size, out_filters,stride, _name,dilationFactor = 1,pad='SAME'):
        with tf.variable_scope(_name):
            output = tf.contrib.layers.separable_conv2d(
                        features,
                        None,
                        kernel_size,
                        depth_multiplier=1,
                        stride=stride,
                        weights_initializer=self.weightInitializer(),
                        normalizer_fn=self.normalizer_fn,
                        normalizer_params=self.norm_params,
                        activation_fn=self.activation,
                        weights_regularizer=self.regularizer_fn,
                        padding=pad,
                        scope='dw'
                        )
            output = tf.contrib.layers.conv2d(
                        output,
                        out_filters, [1, 1],
                        stride=1,
                        weights_initializer=self.weightInitializer(),
                        normalizer_fn=self.normalizer_fn,
                        normalizer_params=self.norm_params,
                        activation_fn=self.activation,
                        weights_regularizer=self.regularizer_fn,
                        scope='pw'
                        )
            return output
    
    def model64(self):
        conv1 = self.conv2d(self.input,8,3,2,"conv1") #32x32
        conv11 = self.conv2d(conv1,3,16,1,"conv11")
        pool1 = self.avgPooling(conv11,2,2) #16x16

        conv2 = self.conv2d(pool1,3,32,1,"conv2")
        pool2 = self.avgPooling(conv2,2,2)

        conv3 = self.conv2d(pool2,3,64,1,"conv3")
        pool3 = self.avgPooling(conv3,2,2)

        conv4 = self.conv2d(pool3,3,128,1,"conv4")
        pool4 = self.avgPooling(conv4,2,2)

        conv5 = self.conv2d(pool4,3,256,1,"conv5")#2x2
        pool5 = self.avgPooling(conv5,2,2)# 1x1

       

        self.featureLayers.append(conv5)
        self.featureLayers.append(pool5)

        predicatedBoxes = []
        predicatedClasses =[]
        predicatedConfs = []
        predictions = []

        for i,feature in enumerate(self.featureLayers):
            cells = self.sizeOfFeatures[i] * self.sizeOfFeatures[i]

            predBox = tf.contrib.layers.conv2d(feature, self.nboxesInFeatures[i] * 3, [1, 1], activation_fn=None, weights_initializer=self.weightInitializer(), scope='loc'+str(i))
            predBox = tf.reshape(predBox, (-1,self.nboxesInFeatures[i] * cells ,3))

            predClass = tf.contrib.layers.conv2d(feature, self.nboxesInFeatures[i] * 2, [1, 1], activation_fn=None, weights_initializer=self.weightInitializer(),scope='class' + str(i))
            predClass = tf.reshape(predClass, (-1, self.nboxesInFeatures[i] * cells, 2))

            predConf = tf.contrib.layers.conv2d(feature, self.nboxesInFeatures[i] * 1, [1, 1], activation_fn=None, weights_initializer=self.weightInitializer(),scope='conf' + str(i))
            predConf = tf.reshape(predConf, (-1, self.nboxesInFeatures[i] * cells, 1))
            
            predicatedBoxes.append(predBox)
            predicatedConfs.append(predConf)
            predicatedClasses.append(predClass)
            predictions.append(tf.contrib.layers.softmax(predClass))

        return tf.concat(predicatedBoxes,axis=1),tf.concat(predicatedClasses,axis=1),tf.concat(predicatedConfs,axis=1), tf.concat(predictions,axis=1)

        
    def createDefaultBoxes(self):
        defaultBoxes = []
        for f, feature in enumerate(self.featureLayers):
            fSize = self.sizeOfFeatures[f]
            for i in range(fSize):
                for j in range(fSize):
                    scales = self.scalesInFeatures[f]
                    for k in range(len(scales)):
                        dBox = [(j + 0.5) / fSize,(i + 0.5) / fSize, scales[k], scales[k]]
                        defaultBoxes.append(dBox)
        return tf.convert_to_tensor(defaultBoxes)
  
net = network()
localizations, classes,confidences, predictions = net.model64()
defaultBoxes = net.createDefaultBoxes()

if __name__=='__main__':    
    with tf.Session() as sess:
        # print(sess.run(defaultBoxes))
        # print(sess.run(defaultBoxes))
        # rf_x, rf_y, eff_stride_x, eff_stride_y, eff_pad_x, eff_pad_y = tf.contrib.receptive_field.compute_receptive_field_from_graph_def(
        #     sess.graph.as_graph_def(), 'input', 'separableConvconv4/Relu6',input_resolution=(64,64))
        # print("receptive",rf_x,rf_y, eff_stride_x, eff_stride_y, eff_pad_x, eff_pad_y)
        writer = tf.summary.FileWriter('./graphs', sess.graph)


 


        

