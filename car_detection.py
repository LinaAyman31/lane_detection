import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip


def loadWeightsLayers():
    weights_path = "yolov3.weights"
    cfg_path = "yolov3.cfg"

    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    names = net.getLayerNames()

    layers_names = [names[i - 1] for i in net.getUnconnectedOutLayers()]

    return layers_names