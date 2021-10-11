#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
# https://github.com/samsu2018/object-detection-opencv
############################################


import cv2
import argparse
import numpy as np
import glob
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help = 'path to input image')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
ap.add_argument('-t', '--type', default= 'single',
                help = 'single or batch image')
args = ap.parse_args()

# global
classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imwrite("object-detection.jpg", img)


def single(img):
    tStart = time.time()
    image = cv2.imread(img)
    
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    net = cv2.dnn.readNet(args.weights, args.config)
    
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
    
    net.setInput(blob)
    
    outs = net.forward(get_output_layers(net))
    
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    results=[]
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], int(round(x)), int(round(y)), int(round(x+w)), int(round(y+h)))
        print("class_ids: {} confidences: {} ".format(class_ids[i], confidences[i]))
#        print(int(round(x)), int(round(y)), int(round(x+w)), int(round(y+h)))
        results.append([class_ids[i], round(confidences[i],4)])
    tEnd = time.time()
    print('{}, objected:{}, {} sec'.format(img, results,tEnd - tStart))


def multi(path):
    for f in glob.glob(os.path.join(path, "*.jpg")):
        single(f)

#cv2.imshow("object detection", image)
#cv2.waitKey()
#    
# cv2.imwrite("object-detection.jpg", image)
#cv2.destroyAllWindows()

# =============================================================================
# Main
# =============================================================================

if args.type=='batch':
    ts = time.time()
    print('Enter batch model')
    multi(args.image)
    te = time.time()
    print("Total cost {} sec".format(ts - te))
else:
    print('Enter single model')
    tStart = time.time()
    single(args.image)
    tEnd = time.time()
#    print("It cost {} sec".format(tEnd - tStart))


