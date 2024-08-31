from imutils import paths
import numpy as np
import imutils
import pickle
import os
import cv2

dataset ="dataset"

embeddingFile="output/embedding.pickle" # initialize the file
embeddingModel="openface-nn4.small2.v1.t7" # initialize the model pytorch

#initialize the caffe model for face modeling 
prototxt ="model/deploy.prototxt"
model="model/res10_300x300_ssd_iter_140000.caffemodel"

#loading caffe model for face modeling
#detecting face from Image via Caffe deep larning model
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

#loading the embedding model
#extracting the embedding via deep learning feature extraction
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

# getting the image paths
imagePaths = list(paths.list_images(dataset))

# initializing the list for embedding
knownEmbedding = []
knownNames = []
total = 0
conf =0.5

#we start to read images  one by one to apply face detection and embedding

for (i,imagePath) in enumerate(imagePaths):
    print("Processing image {}/{}".format(i+1, len(imagePaths)))
    name =imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    #converting the image to blob for face detection
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    #setting the blob for face detection
    detector.setInput(imageBlob)
    #predicting the face detection
    detections = detector.forward()


    if len(detections) > 0:
        i = np.argmax(detections[0,0,:,2])
        confidence = detections[0,0,i,2]

        if confidence > conf:
            #ROI Range of interest
            box = detections[0,0,i,3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]
            (fh, fw) = face.shape[:2]


            if fh <20 or fw <20:
                continue
            #image to blob
            faceBlob = cv2.dnn.blobFromImage(face, 1.0/255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
            #facial features embedder input image face blob 
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            knownNames.append(name)
            knownEmbedding.append(vec.flatten())
            total += 1
print("Embedding:{0}".format(total))
data ={"embeddings":knownEmbedding, "names":knownNames}
f= open(embeddingFile, "wb")
f.write(pickle.dumps(data))
f.close()
print("Processing Completed")

    



