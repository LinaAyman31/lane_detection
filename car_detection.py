import numpy as np
import cv2

def getNet():
    weights_path = "yolov3.weights"
    cfg_path = "yolov3.cfg"

    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return net

def loadWeightsLayers(net):
    names = net.getLayerNames()

    layers_names = [names[i - 1] for i in net.getUnconnectedOutLayers()]

    return layers_names


def netForwardOutput(img ,net ,layers_names):
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (320, 320), crop=False, swapRB=False)

    net.setInput(blob)

    layers_output = net.forward(layers_names)
    return layers_output

def getBoxes(layers_output, img):
    boxes = []
    confidences = []
    classIDs = []
    idxs=[]
    (H, W) = img.shape[:2]
    for output in layers_output:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if (confidence > 0.75):
                
                box = detection[:4] * np.array([W, H, W, H])
                
                bx, by, bw, bh = box.astype("int")
                
                x = int(bx - (bw / 2))
                y = int(by - (bh / 2))
                
                boxes.append([x, y, int(bw), int(bh)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    if len(boxes) != 0:
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.75, 0.5)           
    return idxs, boxes, confidences


   

def drawBoxes(idxs, boxes, confidences, img):
    if len(boxes) != 0:
        for i in idxs.flatten():
            (x, y) = [boxes[i][0], boxes[i][1]]
            (w, h) = [boxes[i][2], boxes[i][3]]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(img, "Car: {}".format(confidences[i]), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return img 


def draw_cars(idxs,boxes,img):
    cars=[]
    num=0 
    if len(boxes) != 0:
        for i in idxs.flatten():
            (x, y) = [boxes[i][0], boxes[i][1]]
            (w, h) = [boxes[i][2], boxes[i][3]]
            if(num<4):
                img1  = img[y:y+h,x:x+w]
                if(img1.shape[0]>50 and img1.shape[1]>50 ):
                    cars.append(cv2.resize(img1,(img.shape[1]//6,img.shape[0]//6)))
                    num+=1
    num=0
    padding =22
    fontScale=1
    thickness=2
    fontFace = cv2.FONT_HERSHEY_SIMPLEX

    for car in cars:
        img[padding*(num+1)+(num*car.shape[0]):padding*(num+1)+((num+1)*car.shape[0]) ,padding:padding+car.shape[1]] =car
        cv2.putText(img, ("car"+str(num+1) ), (padding ,padding*(num+1)+((num)*car.shape[0])), fontFace, fontScale,(255,255,255), thickness,  lineType = cv2.LINE_AA)

        num+=1
    return img


net = getNet()
layers_names = loadWeightsLayers(net)

def carDetection_pipeline_v1(img):
    layers_output = netForwardOutput(img ,net ,layers_names)
    idxs, boxes, confidences = getBoxes(layers_output, img) 
    img = drawBoxes(idxs, boxes, confidences, img)
    return img

def carDetection_pipeline_v2(img):
    layers_output = netForwardOutput(img ,net ,layers_names)
    idxs, boxes, confidences = getBoxes(layers_output, img)
    img = draw_cars(idxs,boxes,img)   
    img = drawBoxes(idxs, boxes, confidences, img)
    return img