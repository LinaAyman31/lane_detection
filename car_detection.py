import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip


def loadWeightsLayers(net):
    names = net.getLayerNames()

    layers_names = [names[i - 1] for i in net.getUnconnectedOutLayers()]

    return layers_names

def getNet():
    weights_path = "yolov3.weights"
    cfg_path = "yolov3.cfg"

    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return net


def netForwardOutput(img ,net ,layers_names):
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), crop=False, swapRB=False)

    net.setInput(blob)

    layers_output = net.forward(layers_names)
    return layers_output

def getBoxes(layers_output, img):
    boxes = []
    confidences = []
    classIDs = []
    (H, W) = img.shape[:2]
    for output in layers_output:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if (confidence > 0.85):
                
                box = detection[:4] * np.array([W, H, W, H])
                
                bx, by, bw, bh = box.astype("int")
                
                x = int(bx - (bw / 2))
                y = int(by - (bh / 2))
                
                boxes.append([x, y, int(bw), int(bh)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    if len(boxes) != 0:
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.5)           
    return idxs, boxes, confidences

def drawBoxes(idxs, boxes, confidences, img):
    if len(boxes) != 0:
        for i in idxs.flatten():
            (x, y) = [boxes[i][0], boxes[i][1]]
            (w, h) = [boxes[i][2], boxes[i][3]]
            
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(img, "Car: {}".format(confidences[i]), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return img 

net = getNet()
layers_names = loadWeightsLayers(net)

def pipeline(img):
    layers_output = netForwardOutput(img ,net ,layers_names)
    idxs, boxes, confidences = getBoxes(layers_output, img)
    img = drawBoxes(idxs, boxes, confidences, img)

    return img