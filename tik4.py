
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
#from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
#from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
#from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os
#from google.colab.patches import cv2_imshow

path = "/truba_scratch/agasi/Dataset/"

INIT_LR = 1e-4
NUM_EPOCHS = 20
BATCH_SIZE = 32

print("[INFO] loading dataset...")
data = []
labels = []
s_bbox = []
o_bbox = []

for csvPath in paths.list_files(path, validExts=(".csv")):
	rows = open(csvPath).read().strip().split("\n")

	for row in rows:
		row = row.split(";")
		(obj, subject, predicate, subx, suby, subw, subh, objx, objy, objw, objh, image_id) = row
		#(_, _, predicate, subx, suby, subw, subh, objx, objy, objw, objh, _) = row
		
		imgPath = os.path.join(["/truba_scratch/agasi/Dataset/images/",image_id+".jpg"])
		image = cv2.imread(imgPath)
		(h, w) = image.shape[:2]
	
		subx = float(subx) / w
		suby = float(suby) / h
		subw = float(subw) / w
		subh = float(subh) / h

		objx = float(objx) / w
		objy = float(objy) / h
		objw = float(objw) / w
		objh = float(objh) / h

		image = load_img(imgPath, target_size=(224, 224))
		image = img_to_array(image)

		data = np.append(data, image)
		labels = np.append(labels, predicate)
		s_bbox = np.append(s_bbox, (subx, suby, subw, subh))
		o_bbox = np.append(o_bbox, (objx, objy, objw, objh))

data = np.array(data, dtype="float32") / 255.0
s_bbox = np.array(s_bbox, dtype="float32")
o_bbox = np.array(o_bbox, dtype="float32")
labels = np.array(labels) 

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

data = np.reshape(data, (-1,224,224,3))
s_bbox = np.reshape(s_bbox, (-1,1,4))
o_bbox = np.reshape(o_bbox, (-1,1,4))

split_data = train_test_split(data,test_size=0.20, random_state=42)
(trainImages, testImages) = split_data[:2]

split_s_bbox = train_test_split(s_bbox,test_size=0.20, random_state=42)
(trainsBBoxes, testsBBoxes) = split_s_bbox[:2]

split_o_bbox = train_test_split(o_bbox,test_size=0.20, random_state=42)
(trainoBBoxes, testoBBoxes) = split_o_bbox[:2]

split_labels = train_test_split(labels,test_size=0.20, random_state=42)
(trainLabels, testLabels) = split_labels[:2]

with mirrored_strategy.scope():
	vgg = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
	vgg.trainable = False

	flatten = vgg.output
	flatten = Flatten()(flatten)

	sbboxHead = Dense(128, activation="relu")(flatten)
	sbboxHead = Dense(64, activation="relu")(sbboxHead)
	sbboxHead = Dense(32, activation="relu")(sbboxHead)
	sbboxHead = Dense(4, activation="sigmoid", name="subject_bbox")(sbboxHead)

	obboxHead = Dense(128, activation="relu")(flatten)
	obboxHead = Dense(64, activation="relu")(obboxHead)
	obboxHead = Dense(32, activation="relu")(obboxHead)
	obboxHead = Dense(4, activation="sigmoid", name="object_bbox")(obboxHead)

	predicate = Dense(512, activation="relu")(flatten)
	predicate = Dropout(0.5)(predicate)
	predicate = Dense(512, activation="relu")(predicate)
	predicate = Dropout(0.5)(predicate)
	predicate = Dense(len(lb.classes_), activation="softmax", name="predicate")(predicate)

	model = Model(inputs=vgg.input, outputs=(sbboxHead, obboxHead, predicate))

	losses = { "predicate": "categorical_crossentropy", "subject_bbox": "mean_squared_error", "object_bbox": "mean_squared_error", }
	lossWeights = { "predicate": 1.0, "subject_bbox": 1.0, "object_bbox": 1.0 }

	opt = Adam(learning_rate=1e-4)
	model.compile(loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights)

print(model.summary())

trainTargets = {"subject_bbox": trainsBBoxes, "object_bbox": trainoBBoxes, "predicate": trainLabels}
testTargets = {"subject_bbox": testsBBoxes, "object_bbox": testoBBoxes, "predicate": testLabels}

print("[INFO] training model...")
H = model.fit(trainImages, trainTargets, validation_data=(testImages, testTargets),batch_size=BATCH_SIZE,epochs=NUM_EPOCHS,verbose=1)

# serialize the model to disk
print("[INFO] saving object detector model...")
model.save("/truba_scratch/agasi/tik4/detector.h5", save_format="h5")

# serialize the label binarizer to disk
print("[INFO] saving label binarizer...")
f = open("/truba_scratch/agasi/trb/lb.pickle", "wb")
f.write(pickle.dumps(lb))
f.close()

print("[INFO] loading object detector...")

model = load_model("/truba_scratch/agasi/trb/detector.h5")
lb = pickle.loads(open("/truba_scratch/agasi/tik4/lb.pickle", "rb").read())

image = load_img("/truba_scratch/agasi/Dataset/images/1.jpg", target_size=(224, 224))
image = img_to_array(image) / 255.0
image = np.expand_dims(image, axis=0)

(s_bboxPreds, o_bboxPreds, prediction) = model.predict(image)
(s_bbox_startX, s_bbox_startY, s_bbox_endX, s_bbox_endY) = s_bboxPreds[0]
(o_bbox_startX, o_bbox_startY, o_bbox_endX, o_bbox_endY) = o_bboxPreds[0]

i = np.argmax(prediction, axis=1)
print(i)
print(lb.classes_)
label = lb.classes_[i][0]
print(label)

print(s_bboxPreds)
print(o_bboxPreds)