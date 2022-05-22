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