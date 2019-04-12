import tensorflow as tf
import numpy as np
import cv2
import glob
import time
import os

sess = tf.Session()

saver = tf.train.import_meta_graph('./models/w5/1.meta')
saver.restore(sess, tf.train.latest_checkpoint('./models/w5/'))
graph = tf.get_default_graph()
input =graph.get_tensor_by_name("input:0")
ist =graph.get_tensor_by_name("isTraning:0")

model =graph.get_tensor_by_name("predictor/predectedBox:0")
conf =graph.get_tensor_by_name("predictor/predictedBoxConf:0")
cla =graph.get_tensor_by_name("predictor/predictedBoxClass:0")
# softmax = tf.nn.softmax(model[:,:2], name='softmax')
# argmax = tf.argmax(softmax, axis=-1, name='argmax', output_type=tf.int64)
nTotal = 0
nDetected = 0
ratio = 0
showAll = True
def getBox(bbox,windowSize):
    x1 = int(bbox[0]*windowSize[0]) - int(bbox[2]*windowSize[0]/2)
    y1 = int(bbox[1]*windowSize[1]) - int(bbox[3]*windowSize[1]/2)
    w = int(bbox[2]*windowSize[0])
    h = int(bbox[3]*windowSize[1])
    x2 = int(bbox[0]*windowSize[0]) + int(bbox[2]*windowSize[0]/2)
    y2 = int(bbox[1]*windowSize[1]) + int(bbox[3]*windowSize[1]/2)
    return (x1,y1),(x2,y2)

stepSize = 64
# f= open("./fineTune/bbBoxs.txt","w+")
for fileAddress in glob.glob('./dataSet/neg/*'):
    if fileAddress.split('.')[-1] == 'png' or fileAddress.split('.')[-1] == 'jpg':
        img = cv2.imread(fileAddress)
        # img = cv2.resize(img, (32, 32)) 
        nTotal =nTotal+1
        print(len(img),len(img[0]))
        # for y in range(0, img.shape[0], stepSize):
            # for x in range(0, img.shape[1], stepSize):
        windowSize = [len(img[0]),len(img)]
        i =img
        i = cv2.resize(i, (64, 64))
        i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        i = np.reshape(i,(64,64,1))

        i = (i - i.mean()) / i.std()
        
        start = time.time()
        _model,_conf,_cla = sess.run([model,conf,cla], feed_dict={input: [i],ist:False})
        end = time.time()
        print(_model)

        out = _cla[0,1]
        # prob = prob[0]
        
        bbox = _model[0,:]
        b1p1, b1p2 = getBox(bbox,windowSize)
         

        if out > 0.5:
            # cv2.rectangle(img,(x,y),(x+windowSize,y+windowSize),(255,0,0),1)
            # os.rename(fileAddress, "./pos/"+fileAddress.split("/")[-1])
            # f.write(fileAddress.split("/")[-1]+" "+str(int(bbox[0]*windowSize))+" "+str(int(bbox[1]*windowSize))+" "+str(int(w/2.0))+"\n")
            # mostConf = np.argmax(conf)
            # print('confince of prediction : ',float(conf[0]),float(conf[1]),float(conf[2]),float(conf[3]))
            # if showAll == True:
            print('conf: ',_conf,'class prob: ',_cla)
            cv2.rectangle(img,b1p1,b1p2,(255,0,0),1)
            #     cv2.rectangle(img,b1p1,b1p2,(255,0,0),1)
            #     cv2.rectangle(img,b3p1,b3p2,(0,0,255),1)
            #     cv2.rectangle(img,b4p1,b4p2,(255,255,0),1)
            # else:
            #     if mostConf == 0:
            #         cv2.rectangle(img,b1p1,b1p2,(255,0,0),1)
            #     elif mostConf == 1:
            #         cv2.rectangle(img,b2p1,b2p2,(0,255,0),1)
            #     elif mostConf == 2:
            #         cv2.rectangle(img,b3p1,b3p2,(0,0,255),1)
            #     elif mostConf == 3:
            #         cv2.rectangle(img,b4p1,b4p2,(255,255,0),1)
            # # cv2.line(img,(int(bbox[0]*32),int(bbox[1]*32)),(int(bbox[0]*32),int(bbox[1]*32)),(255,0,0),4)
            nDetected = nDetected +1
            # cv2.imshow('img',img)
            # cv2.waitKey(0)
        
ratio = nDetected / float(nTotal)
print("ratio",ratio, nDetected,nTotal)

    # print(out.shape)
    