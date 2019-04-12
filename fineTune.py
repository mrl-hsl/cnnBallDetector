import tensorflow as tf
import glob
import os
import cv2
# import network
import dataManager as data
import numpy as np
class teacher():
    def __init__(self,i):
        # self.model = network.model
        # self.input = network.net.input
        # self.isTraining = network.net.isTraning
        self.i = i
        self.sess = tf.Session()

        saver = tf.train.import_meta_graph('./models/f11/1.meta')
        saver.restore(self.sess, tf.train.latest_checkpoint('./models/f11/'))
        graph = tf.get_default_graph()

        self.bbox = graph.get_tensor_by_name('bbox:0')
        self.label = graph.get_tensor_by_name('label:0')
        self.lrate = graph.get_tensor_by_name('learning_rate:0')
        # self.global_step = tf.Variable(0, trainable=False, name='global_step')
        
        
        self.input =graph.get_tensor_by_name("input:0")
        self.isTraining =graph.get_tensor_by_name("isTraning:0")

        self.model =graph.get_tensor_by_name("model:0")
        self.graph = graph
        self.trainBatch = data.next_train_batch
        self.trainIt = data.train_iterator
        
        self.testBatch = data.next_test_batch
        self.testIt = data.test_iterator

        self.buildCostFunction()
        self.loss = tf.losses.get_total_loss()
        tf.summary.scalar('total_loss', self.loss)
        self.buildOptimizer()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.train_op = tf.group([self.train_op, update_ops])

        # self.loss, self.optimizer, self.train_op = self.buildOptimizer()
        self.buildPredector()
        self.buildMetrics()

        


        self.train_writer = tf.summary.FileWriter('logs/' + str(i) + "/train", self.sess.graph, flush_secs=10)
        self.test_writer = tf.summary.FileWriter('logs/' + str(i) + "/test", self.sess.graph, flush_secs=10)
        self.saver = tf.train.Saver()
        # self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.trainIt.initializer)
        self.sess.run(self.testIt.initializer)
        self.sess.run(tf.local_variables_initializer())
        # self.graph = tf.get_default_graph()
        self.merged = tf.summary.merge_all()
    def buildCostFunction(self):
        with tf.name_scope('cost'):
            bbox = self.model[:,2:]
            label = self.model[:,:2]

            bboxDelta = bbox - self.bbox
            bboxSum = tf.reduce_sum(tf.multiply(tf.reduce_sum(tf.square(bboxDelta), axis = 1),self.label[:,1]))
            bboxLose = bboxSum / (tf.reduce_sum(self.label[:,1])+0.00001)
            labelLose = tf.losses.softmax_cross_entropy(onehot_labels=self.label, logits=label)
            
            tf.losses.add_loss(bboxLose)
            tf.losses.add_loss(labelLose)
            tf.summary.scalar('bboxLose', bboxLose)
            tf.summary.scalar('labelLose', labelLose)
            tf.summary.scalar('nBalls', tf.reduce_sum(self.label[:,1]))
    def evalOnFolder(self,address):
        nDetected = 0
        nTotal = 0
        for fileAddress in glob.glob(address):
            if nTotal == 4000:
                break
            img = cv2.imread(fileAddress)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (32, 32)) 
            nTotal =nTotal+1
            i =img
            # i = (i - i.mean()) / i.std()

            # start = time.time()
            out,prob,_model = self.sess.run([self.predictions_argmax,self.predictions_argmax,self.model], feed_dict={self.input: [i],self.isTraining:False})
            # end = time.time()
            # print(end - start)

            out = out[0]
            prob = prob[0]
            bbox  = _model[0][2:]
            x1 = int(bbox[0]*32) - int(bbox[2]*16)
            y1 = int(bbox[1]*32) - int(bbox[3]*16)
            w = int(bbox[2]*32)
            h = int(bbox[3]*32)
            x2 = int(bbox[0]*32) + int(bbox[2]*16)
            y2 = int(bbox[1]*32) + int(bbox[3]*16)
            # print(out,prob,bbox,w,h)
            if out == 1:
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),1)
                cv2.line(img,(int(bbox[0]*32),int(bbox[1]*32)),(int(bbox[0]*32),int(bbox[1]*32)),(255,0,0),4)
                nDetected = nDetected +1
                # cv2.imshow('img',img)
                # cv2.waitKey(0)
        return nDetected / float(nTotal)
        

    def buildOptimizer(self):
        # self.optimizer = self.graph.get_operation_by_name("adam_optimizer")
        # tf.train.AdamOptimizer(learning_rate=self.lrate, name='adam_optimizer')
        self.train_op = self.graph.get_operation_by_name("train_op")
        # self.optimizer.minimize(self.loss, global_step=self.global_step, name='train_op')
    def calculatIOU(self):
        with tf.name_scope('IOU'):
            bbox = self.model[:,2:]
            xc, yc, w, h = tf.split(bbox, 4, axis=1)
            lxc, lyc, lw, lh = tf.split(self.bbox, 4, axis=1)
            x1 = xc-(w/2.0)
            y1 = yc-(h/2.0)
            x2 = xc+(w/2.0)
            y2 = yc+(h/2.0)

            lx1 = lxc-(lw/2.0)
            ly1 = lyc-(lh/2.0)
            lx2 = lxc+(lw/2.0)
            ly2 = lyc+(lh/2.0)

            xI1 = tf.maximum(x1, lx1)
            yI1 = tf.maximum(y1, ly1)
            xI2 = tf.minimum(x2, lx2)
            yI2 = tf.minimum(y2, ly2)

            areaIntersection = tf.maximum(0.0,(xI2-xI1)) * tf.maximum(0.0,(yI2-yI1))
            areaLabel = lw*lh
            areaPredected = w*h
            IOU = areaIntersection / ((areaLabel + areaPredected) - areaIntersection)
            # IOU = tf.multiply(IOU, self.label[:,1])
            IOU = tf.reduce_sum(IOU)
            self.iou = IOU / (tf.reduce_sum(self.label[:,1])+0.001)
            tf.summary.scalar('miou', self.iou)
             
            


        
        

    def buildPredector(self):
        with tf.name_scope('predictor'):
            self.softmax_output = tf.nn.softmax(self.model[:,:2], name='softmax_output')
            self.predictions_argmax = tf.argmax(self.softmax_output, axis=-1, name='predictions_argmax', output_type=tf.int64)
    def buildMetrics(self):
        with tf.variable_scope('metrics') as scope:
            self.labels_argmax = tf.argmax(self.label, axis=-1, name='labels_argmax', output_type=tf.int64)
            self.acc_value = tf.reduce_sum(tf.cast(tf.equal(self.labels_argmax,self.predictions_argmax),tf.float32)) / tf.cast(tf.shape(self.label)[0],tf.float32)

            
            # self.acc_value, self.acc_update_op = tf.metrics.accuracy(labels=self.labels_argmax,predictions=self.predictions_argmax)
            # self.acc_value = tf.identity(self.acc_value, name='acc_value')
            # self.acc_update_op = tf.identity(self.acc_update_op, name='acc_update_op')
            # self.local_metric_vars = tf.contrib.framework.get_variables(scope=scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
            # self.metrics_reset_op = tf.variables_initializer(var_list=self.local_metric_vars, name='metrics_reset_op')
            self.accuracy = tf.summary.scalar('accuracy', self.acc_value)
            self.calculatIOU()

    def eval(self,counter):
        try:
            b =self.sess.run(self.testBatch)
            feed={self.input: b[:][0], self.label: b[:][1],self.bbox: np.transpose(b[:][2:]),self.lrate:0.0001,self.isTraining:False}
            summary ,valLose ,_acc_value = self.sess.run([self.merged, self.loss, self.acc_value],feed_dict=feed)
            self.test_writer.add_summary(summary, counter)
            print(counter, "**************, ",valLose,_acc_value)
            
        except tf.errors.OutOfRangeError:
            self.sess.run(self.testIt.initializer)    
            

    def teach(self):
        counter = 0
        while True:
            try:
                # if counter == 10002:
                    # break
                b =self.sess.run(self.trainBatch)
                feed={self.input: b[0][:], self.label: b[1][:],self.bbox: np.transpose(b[2:]),self.lrate:0.001,self.isTraining:True}
    
                summary, opt = self.sess.run([self.merged, self.train_op,], feed_dict=feed)
                self.train_writer.add_summary(summary, counter)
                summary ,valLose ,_acc_value ,meanIou = self.sess.run([self.merged, self.loss, self.acc_value,self.iou], feed_dict=feed)
                
                print(counter, "valLoss, ",valLose,_acc_value,meanIou)
                # if _acc_value > 0.96:
                    # self.saver.save(self.sess, './models/11/1')
                if int((counter+1) % 500) ==0:
                    self.saver.save(self.sess, './models/'+str(self.i)+'/1')
                    break
                    # ratFals = self.evalOnFolder('./dataSet/neg/*')
                    # ratPos = self.evalOnFolder('../../ballTrain/opencv-hog-classifier-training/positive_images/*')
                    # tf.summary.scalar('rateFals', ratFals)
                    # tf.summary.scalar('ratPos', ratPos)
                    # print("rateFalse = ",ratFals,"ratPos = ",ratPos)
                # if int(counter % 10) == 0:
                    # self.eval(counter)
                counter = counter + 1
            except tf.errors.OutOfRangeError:
                self.sess.run(self.trainIt.initializer)

master = teacher('f12')
master.teach()