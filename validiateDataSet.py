import tensorflow as tf
import numpy as np
import cv2
import dataManager as data

getBatch = data.next_test_batch
sess = tf.Session()
sess.run(data.test_iterator.initializer)



def calculatIOU(boxes1,boxes2):
    xc, yc, w, h = np.split(boxes1, 4, axis=-1)
    lxc, lyc, lw, lh = np.split(boxes2, 4, axis=-1)
    x1 = xc-(w/2.0)
    y1 = yc-(h/2.0)
    x2 = xc+(w/2.0)
    y2 = yc+(h/2.0)

    lx1 = lxc-(lw/2.0)
    ly1 = lyc-(lh/2.0)
    lx2 = lxc+(lw/2.0)
    ly2 = lyc+(lh/2.0)

    xI1 = np.maximum(x1, lx1)
    yI1 = np.maximum(y1, ly1)
    xI2 = np.minimum(x2, lx2)
    yI2 = np.minimum(y2, ly2)

    areaIntersection = np.maximum(0.0,(xI2-xI1)) * np.maximum(0.0,(yI2-yI1))
    areaLabel = lw*lh
    areaPredected = w*h
    return np.divide(areaIntersection , ((areaLabel + areaPredected) - areaIntersection))

saver = tf.train.import_meta_graph('./4-8-depthwise/1.meta')
saver.restore(sess, tf.train.latest_checkpoint('./4-8-depthwise/'))
graph = tf.get_default_graph()
input =graph.get_tensor_by_name("input:0")
ist =graph.get_tensor_by_name("isTraning:0")

bboxesTensor =graph.get_tensor_by_name("predictor/predectedBox:0")
confTensor =graph.get_tensor_by_name("predictor/predictedBoxConf:0")
classTensor =graph.get_tensor_by_name("predictor/predictedBoxClass:0")

accuracy = 0.0
nTP = 0.0
nFP = 0.0
nTN = 0.0
nFN = 0.0
tP = 0.0
tN = 1.0
nIou5 = 0.0
nIou9 = 0.0
nIou7 = 0.0
sumIou = 0.0
run_meta = tf.RunMetadata()

while True:
    try:
        batch = sess.run(getBatch)
        i = (batch[0][:] - batch[0][:].mean()) / batch[0][:].std()
        bboxes,conf,classProb = sess.run([bboxesTensor,confTensor,classTensor], feed_dict={input: i ,ist:False})
        detectedClass = np.argmax(classProb,axis=1)
        label_class = np.argmax(batch[1][:],axis=1)
        label_bbox =np.asarray([batch[2][:],batch[3][:],batch[4][:],batch[5][:]])
        # print(bboxes.shape,np.transpose(label_bbox).shape)
        iou = calculatIOU(bboxes,np.transpose(label_bbox))
        # print("iou shape", iou.shape)
        # print(label_class.shape)
        for i in range(batch[0].shape[0]):
            # if batch[1][i][1]==1:
            label_bbox = [batch[2][i],batch[3][i],batch[4][i],batch[5][i]]
            b1p1, b1p2 = data.getBox(label_bbox,[64,64])
            pb1p1, pb1p2 = data.getBox(bboxes[i],[64,64])
            if detectedClass[i]==1 and label_class[i]==1:
                nTP += 1
                if iou[i] > 0.9: 
                    nIou9 += 1
                    nIou5 += 1
                    nIou7 += 1
                elif iou[i] > 0.75: 
                    nIou5 += 1
                    nIou7 += 1
                    # img = batch[0][i]
                    # img = img.astype(np.uint8)
                    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    # cv2.rectangle(img,b1p1,b1p2,(255,0,0),1)
                    # cv2.rectangle(img,pb1p1,pb1p2,(0,0,255),1)
                    # cv2.imshow('ss',img)
                    # cv2.waitKey(0)
                elif iou[i] > 0.5: 
                    nIou5 += 1
                sumIou += iou[i]

            if detectedClass[i]==0 and label_class[i]==1:
                nFN += 1
            if detectedClass[i]==0 and label_class[i]==0:
                nTN += 1 
            if detectedClass[i]==1 and label_class[i]==0:
                # print("alo")
                nFP += 1
            if label_class[i] == 0:
                tN += 1
            if label_class[i] == 1:
                tP += 1
            
            



            # else:
            # img = batch[0][i]
            # img = (img*255)
            # img = img.astype(np.uint8)
            # cv2.rectangle(img,b1p1,b1p2,(255,0,0),1)
            # cv2.imshow('ss',img)
            # cv2.waitKey(0)
            # accuracy = accuracy +1
    except tf.errors.OutOfRangeError:
        # g = tf.Graph()
        # tf.profiler.profile(sess.graph,run_meta=run_meta, cmd='op',options=tf.profiler.ProfileOptionBuilder.float_operation())
        break
print('accuracy : ',(nTN+nTP)/(nTN+nTP+nFN+nFP),'\nRecall : ',nTP/(nTP+nFN),'\nPrecsion : ', nTP/(nTP+nFP) ,'\nFalseRate : ', (nFP/tN)  )
print('meanIou : ',sumIou/nTP,'\nAP0.9 : ',nIou9/(nTP+nFP),'\nAp0.7 : ', nIou7/(nTP+nFP),'\nAp0.5 : ', nIou5/(nTP+nFP) )
  
