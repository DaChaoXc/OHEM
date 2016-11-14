import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from utils.cython_nms import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse



def nmsly(dets, nms_thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        #ovr = inter /np.minimum(areas[i], areas[order[1:]])
        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]

    return keep

def demo(net, image_name, CONF_THRESH, nms_thresh, fileout,num, savepath):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load pre-computed Selected Search object proposals
    box_file=image_name+'.mat'
    obj_proposals = sio.loadmat(box_file)['bboxes'] 
    print  obj_proposals

    box_list = []
    for i in xrange(obj_proposals.shape[0]):
        box_list.append(obj_proposals[i, (1, 0, 3, 2)] - 1)  
    box_list_npy = np.array(box_list)
    
    im_file=image_name+'.png'
    im = cv2.imread(im_file)
    print im_file
    # Detect all object classes and regress object bounds
    #_t['im_detect'].tic()
    scores, boxes = im_detect(net, im, box_list_npy)
    #_t['im_detect'].toc()
   
    
    cls_boxes = boxes[:, 4:8]
    cls_scores = scores[:, 1]

    keep_ = np.where(cls_scores >= CONF_THRESH)[0]
    cls_boxes= cls_boxes[keep_, :]
    cls_scores=cls_scores[keep_]
    
    
    dets = np.hstack((cls_boxes,
                       cls_scores[:, np.newaxis])).astype(np.float32)

    keep_S = nmsly(dets, nms_thresh)
    dets= dets[keep_S, :]
           
    inds = np.where(dets[:, -1] >= 0)[0]
    for i in inds:
        bbox = dets[i, :4]							
        score = dets[i, -1]       
        text="%f "%score
        fileout.write("%d,%d,%d,%d,%d,%f\n" %(num,bbox[0],bbox[1],bbox[2]-bbox[0]+1,bbox[3]-bbox[1]+1,score))
        



if __name__ == '__main__':
    
    prototxt="/home/vision/xc/fast-rcnn/models/VGG16/test.prototxt"
       
    filepath="/home/vision/xc/fast-rcnn/data/INRIA_test/INRIA.txt"

    CONF_THRESH = 0.8
    nms_thresh = 0.3
    all_caffemodel = np.arange(1000,37000,1000)

    for caffemodel_num in all_caffemodel:  
       
        fileoutpath="/home/vision/xc/ohem/output/INRIA_test/V000_%d.txt"%caffemodel_num ####

        caffemodel="/home/vision/xc/ohem/output/fast_rcnn_ohem/my_trainval/vgg16_fast_rcnn_iter_%d.caffemodel"%caffemodel_num

        print caffemodel


        if os.path.exists(filepath.split('.')[0])==False:
           os.mkdir(filepath.split('.')[0]) 
   
        caffe.set_mode_gpu()
        caffe.set_device(0)
    
        fileout=open(fileoutpath,"w")

        net = caffe.Net(prototxt, caffemodel, caffe.TEST)
   
        print '``````````````now deal with model number:%d````````````````'%caffemodel_num ####

        with open(filepath) as f:
            split_line=[line.strip() for line in f.readlines()]
   
        num=1
        for line in split_line:          
            image_name='/home/vision/xc/fast-rcnn/data/INRIA_test/test/'+line  
            demo(net, image_name, CONF_THRESH, nms_thresh, fileout, num, filepath)   
            num=num+1
 
        fileout.close()





