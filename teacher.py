import tensorflow as tf
import glob
import os
import cv2
import network
import dataManager as data
from tf_extended import math as tfe_math

import numpy as np
class teacher():
    def __init__(self,i):
        self.localizations = network.localizations
        self.classes = network.classes
        self.confidences = network.confidences
        self.predictions = network.predictions
        self.input = network.net.input
        self.isTraining = network.net.isTraning
        self.defaultBoxes = network.defaultBoxes

        self.loc_loss_fn = tf.abs
        self.conf_loss_fn = tf.abs

        self.negetiveBoxUsageRate = 3.0
        self.i = i
        self.bbox = tf.placeholder(tf.float32, [None, 4], name='bbox')
        self.label = tf.placeholder(tf.float32, [None, 2], name='label')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.lrate = tf.train.piecewise_constant(self.global_step, [5000,10000],[1e-3,1e-4,1e-5])
        tf.summary.scalar('lerning_rate',self.lrate)

        self.trainBatch = data.next_train_batch
        self.trainIt = data.train_iterator

        self.testBatch = data.next_test_batch
        self.testIt = data.test_iterator

        self.buildCostFunction()
        self.loss = tf.losses.get_total_loss()
        self.buildOptimizer()

        self.buildPredector()
        self.buildMetrics()

        self.sess = tf.Session()


        self.train_writer = tf.summary.FileWriter('logs/' + str(i) + "/train", self.sess.graph, flush_secs=10)
        self.test_writer = tf.summary.FileWriter('logs/' + str(i) + "/test", self.sess.graph, flush_secs=10)
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.trainIt.initializer)
        self.sess.run(self.testIt.initializer)
        self.sess.run(tf.local_variables_initializer())
        self.graph = tf.get_default_graph()
        self.merged = tf.summary.merge_all()



    def matchBoxes(self):
        with tf.name_scope("matching"):
            matchThreshood = 0.5
            i=0
            nDefaultBoxes = self.defaultBoxes.get_shape().as_list()[0]
            matchedIous = tf.TensorArray(tf.float32, size=nDefaultBoxes, dynamic_size=False, infer_shape=True)

            def condition(i,match):
                return tf.less(i,nDefaultBoxes)

            def m_body(i,match):
                jaccard = self.calculatIOU(tf.reshape(self.defaultBoxes[i,:],(1,4)),self.bbox)
                match = match.write(i,jaccard)
                return [i+1,match]

            [i,matchedIous] = tf.while_loop(condition,m_body,[i,matchedIous])

            matchedIous = tf.transpose(matchedIous.stack())
            bestMatchMask = tf.one_hot(tf.argmax(matchedIous,axis=1),nDefaultBoxes,on_value=True,off_value=False)
            goodMatchMask = tf.greater(matchedIous,matchThreshood)
            matchMask = tf.cast(tf.logical_or(bestMatchMask,goodMatchMask),tf.float32)
            
            posMatchMask = tf.transpose(tf.multiply(tf.transpose(matchMask), self.label[:, 1]))

            return posMatchMask

    def abs_smooth(self,x):

        absx = tf.abs(x)
        minx = tf.minimum(absx, 1)
        r = 0.5 * ((absx - 1) * minx + absx)
        return r

    def calculate_localization_loss(self,selectedBoxes,nSelectedBoxes,predicatedBoxes):
        with tf.name_scope("localization_cost"):
            nSelectedBoxes = tf.cast(nSelectedBoxes, tf.int32)
            gBoxesInSelectedBoxes = tf.TensorArray(tf.float32, size=nSelectedBoxes, dynamic_size=False,infer_shape=True)
            selectedPridictedBoxes = tf.TensorArray(tf.float32, size=nSelectedBoxes, dynamic_size=False,infer_shape=True)
            i = 0

            def condition(i,gBoxInDBox,predBox):
                return i<nSelectedBoxes

            def body(i,gBoxInDBox,predBox):
                batchIndx , dBoxIndx = tf.unstack(selectedBoxes[i,:])

                gx , gy, gw, gh = tf.unstack(self.bbox[batchIndx,:])
                dbx , dby, dbw, dbh = tf.unstack(self.defaultBoxes[dBoxIndx,:])

                gdx = (gx - dbx) / dbw
                gdy = (gy - dby) / dbh
                gdw = tf.log(gw / dbw)
                gdh = tf.log(gh / dbh)

                gdBox = tf.stack([gdx,gdy,gdw])
                gBoxInDBox = gBoxInDBox.write(i, gdBox)

                predBox = predBox.write(i,predicatedBoxes[batchIndx,dBoxIndx,:])
                return [i+1,gBoxInDBox, predBox]

            [i,gBoxesInSelectedBoxes,selectedPridictedBoxes] = tf.while_loop(condition,body,[i,gBoxesInSelectedBoxes,selectedPridictedBoxes]);

            selectedPridictedBoxes = selectedPridictedBoxes.stack()
            gBoxesInSelectedBoxes = gBoxesInSelectedBoxes.stack()

            loc = tf.reduce_sum(self.loc_loss_fn(tf.subtract(selectedPridictedBoxes, gBoxesInSelectedBoxes)))

            return loc 

    def calculate_confidence_loss(self,selectedBoxes,nSelectedBoxes,predicatedConf,predicatedBoxes,notSelectedBoxes, nNegetiveUsage):
        with tf.name_scope("conf_Cost"):
            i = 0
            nSelectedBoxes = tf.cast(nSelectedBoxes,tf.int32)
            selectedPridictedIou = tf.TensorArray(tf.float32, size=nSelectedBoxes, dynamic_size=False, infer_shape=True);
            selectedPridictedConf = tf.TensorArray(tf.float32, size=nSelectedBoxes, dynamic_size=False, infer_shape=True);

            def condition(i, predIou, predConf):
                return i < nSelectedBoxes

            def body(i, predIou, predConf):
                batchIndx, dBoxIndx = tf.unstack(selectedBoxes[i, :])
                
                pbx, pby, pbr= tf.unstack(predicatedBoxes[batchIndx,dBoxIndx,:])
                pbw = pbr
                pbh = pbw

                dbx, dby, dbw, dbh = tf.unstack(self.defaultBoxes[dBoxIndx,:])

                x = pbx * dbw + dbx
                y = pby * dbh + dby
                w = tf.exp(pbw) * dbw
                h = tf.exp(pbh) * dbh

                decodedPeredicatedBox = tf.stack([x,y,w,h])

                iou = self.calculatIOU(self.bbox[batchIndx,:],decodedPeredicatedBox)
                predIou = predIou.write(i,iou)
                predConf = predConf.write(i, predicatedConf[batchIndx,dBoxIndx])
                return [i + 1, predIou, predConf]

            [i, selectedPridictedIou,selectedPridictedConf] = tf.while_loop(condition, body, [i, selectedPridictedIou,selectedPridictedConf]);

            selectedPridictedIou = selectedPridictedIou.stack()
            selectedPridictedConf = selectedPridictedConf.stack()

            posLoss = tf.reduce_sum((self.conf_loss_fn(selectedPridictedConf-selectedPridictedIou)))

            negConfs = tf.squeeze(tf.gather_nd(predicatedConf, notSelectedBoxes))

            topNegConfs, index = tf.nn.top_k(negConfs, nNegetiveUsage)

            negLoss = tf.reduce_sum(self.conf_loss_fn(topNegConfs))



            return posLoss + negLoss

    def calculate_class_loss(self, selectedBoxes, nSelectedBoxes, predictedClasses,notSelectedBoxes,nNegetiveUsage):

        nSelectedBoxes = tf.cast(nSelectedBoxes, tf.int32)
        posConfs = tf.nn.softmax(tf.gather_nd(predictedClasses, selectedBoxes))
        posLoss = - tf.reduce_sum(tf.log(tf.clip_by_value(posConfs[:,1],1e-8,1.0)))

        negScores = tf.nn.softmax(tf.gather_nd(predictedClasses,notSelectedBoxes))
        negScores = negScores[:,0]
        topNegScores,index = tf.nn.top_k(-negScores,k= nNegetiveUsage)

        negLoss = - tf.reduce_sum(tf.log(tf.clip_by_value(-topNegScores,1e-8,1.0)))

        return posLoss + negLoss



    def buildCostFunction(self):
        with tf.name_scope('cost'):
            
            alpha = 5.0

            matchMask = self.matchBoxes()

            predBoxes, predClasses,predConf = self.localizations , self.classes, self.confidences
            nBoxes = predBoxes.get_shape().as_list()[1]

            selectedBoxes = tf.where(tf.cast(matchMask, tf.bool))
            notSelectedBoxes = tf.where(tf.logical_not(tf.cast(matchMask,tf.bool)))
            nSelectedBoxes = tf.reduce_sum(matchMask)
            nNotSelectedBoxes = tf.shape(notSelectedBoxes)[0]

            nNegetiveUsage = tf.cast(tf.minimum(tf.cast(nNotSelectedBoxes,tf.float32), self.negetiveBoxUsageRate * nSelectedBoxes),tf.int32)
            nNegetiveUsage = tf.maximum(nNegetiveUsage, 1) #FIX TOP_K



            localizationLoss = self.calculate_localization_loss(selectedBoxes, nSelectedBoxes,predBoxes)
            confidenceLoss = self.calculate_confidence_loss(selectedBoxes, nSelectedBoxes, predConf,predBoxes,notSelectedBoxes,nNegetiveUsage)
            classLoss = self.calculate_class_loss(selectedBoxes, nSelectedBoxes, predClasses,notSelectedBoxes,nNegetiveUsage)

            confidenceLoss = tf.cond(nSelectedBoxes > 0, lambda : confidenceLoss, lambda : 0.0)
            classLoss = tf.cond(nSelectedBoxes > 0, lambda : classLoss, lambda : 0.0)

            useClassLoss =tf.maximum(tf.cast(tf.less(self.global_step,7000),tf.float32)*0.6,0.2)
            loss = (useClassLoss * classLoss) + (alpha * localizationLoss) + confidenceLoss
            loss = tfe_math.safe_divide(loss, nSelectedBoxes)
            self.test = tf.identity(tf.reduce_sum(self.label[:,1]))
            tf.losses.add_loss(loss)

            tf.summary.scalar('locLoss', localizationLoss)
            tf.summary.scalar('confLoss', confidenceLoss)
            tf.summary.scalar('classLoss', classLoss)
            tf.summary.scalar('totalLos', loss)

    def buildOptimizer(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate= self.lrate, name='adam_optimizer')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gradients = self.optimizer.compute_gradients(self.loss)
            self.train_op = self.optimizer.apply_gradients(gradients, global_step=self.global_step, name='train_op')

    def calculatIOU_single(self,boxes1,boxes2):
        with tf.name_scope('IOU'):
            xc, yc, w, h = tf.split(boxes1, 4, axis=0)
            lxc, lyc, lw, lh = tf.split(boxes2, 4, axis=0)
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
            IOU = tfe_math.safe_divide(areaIntersection , ((areaLabel + areaPredected) - areaIntersection))

            return tf.reduce_sum(IOU)

    def calculatIOU(self,boxes1,boxes2):
        with tf.name_scope('IOU'):
            xc, yc, w, h = tf.split(boxes1, 4, axis=-1)
            lxc, lyc, lw, lh = tf.split(boxes2, 4, axis=-1)
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
            IOU = tfe_math.safe_divide(areaIntersection , ((areaLabel + areaPredected) - areaIntersection))

            return tf.reduce_sum(IOU,axis=-1)
    
    def buildPredector(self):
        with tf.name_scope('predictor'):
            predBoxes, predClasses,predConf = self.localizations , self.classes, self.confidences
            nBoxes = predBoxes.get_shape().as_list()[1]

            posClassProb = tf.reshape(self.predictions[:,:,1],(-1,nBoxes))
            iouConf = tf.reshape(predConf,(-1,nBoxes))
            maxConfIndx = tf.argmax(tf.multiply(posClassProb, iouConf), axis=1)
            batchIndx = tf.range(0,tf.cast(tf.shape(predConf)[0],tf.int64),1,dtype=tf.int64)
            batchConfIndx = tf.stack([batchIndx,maxConfIndx],axis=1)

            encodedPredectedBox = tf.gather_nd(predBoxes,batchConfIndx)
            predictedBoxConf = tf.gather_nd(predConf,batchConfIndx)
            predictedBoxConf = tf.identity(predictedBoxConf, name='predictedBoxConf')
            predictedBoxClass = tf.gather_nd(self.predictions,batchConfIndx)
            self.predictedBoxClass = tf.identity(predictedBoxClass, name='predictedBoxClass')

            bestDefaultBoxes = tf.gather(self.defaultBoxes,maxConfIndx,axis=0)

            pbx, pby, pbr = tf.unstack(encodedPredectedBox,axis=1)
            pbw = pbr
            pbh = pbw

            dbx, dby, dbw, dbh = tf.unstack(bestDefaultBoxes,axis=1)

            x = (pbx * dbw) + dbx
            y = (pby * dbh) + dby
            w = tf.exp(pbw) * dbw
            h = tf.exp(pbh) * dbh

            self.predectedBox = tf.stack([x,y,w,h],axis=1)
            self.predectedBox = tf.identity(self.predectedBox,name = 'predectedBox')


    def meanIou(self):
        IOU = self.calculatIOU(self.bbox,self.predectedBox)

        IOU = tf.multiply(IOU, self.label[:,1])
        IOU = tf.reduce_sum(IOU,axis=0)
        self.iou = tfe_math.safe_divide(IOU , tf.reduce_sum(self.label[:,1]))
        tf.summary.scalar('miou', self.iou)


    def buildMetrics(self):
        with tf.variable_scope('metrics') as scope:
            self.labels_argmax = tf.argmax(self.label, axis=-1, name='labels_argmax', output_type=tf.int64)
            self.predictedBoxClass_argmax = tf.argmax(self.predictedBoxClass, axis=-1, name='predictedBoxClass_argmax', output_type=tf.int64)
            self.acc_value = tf.reduce_sum(tf.cast(tf.equal(self.labels_argmax,self.predictedBoxClass_argmax),tf.float32)) / tf.cast(tf.shape(self.label)[0],tf.float32)
            tf.summary.scalar('acc', self.acc_value)
            self.meanIou()
    def eval(self,counter):
        try:
            b =self.sess.run(self.testBatch)
            feed={self.input: b[:][0], self.label: b[:][1],self.bbox: np.transpose(b[:][2:]),self.isTraining:False}
            summary ,valLose,_p  = self.sess.run([self.merged, self.loss,self.predectedBox],feed_dict=feed)
            self.test_writer.add_summary(summary, counter)
            print(counter, "**************, ",valLose)

        except tf.errors.OutOfRangeError:
            self.sess.run(self.testIt.initializer)


    def teach(self):
        counter = 0
        while True:
            try:
                b =self.sess.run(self.trainBatch)
                feed={self.input: b[0][:], self.label: b[1][:],self.bbox: np.transpose(b[2:]),self.isTraining:True}
                opt = self.sess.run([self.train_op,], feed_dict=feed)
                summary ,valLose,p  = self.sess.run([self.merged, self.loss,self.predectedBox], feed_dict=feed)
                self.train_writer.add_summary(summary, counter)
                if counter % 50 == 0:
                    self.eval(counter)


                print(counter)
                if int((counter+1) % 500) ==0:
                    self.saver.save(self.sess, './models/'+str(self.i)+'/1')
                    self.train_writer.flush()
                counter = counter + 1
            except tf.errors.OutOfRangeError:
                self.sess.run(self.trainIt.initializer)

master = teacher('8-16-noDepthWise')
master.teach()
