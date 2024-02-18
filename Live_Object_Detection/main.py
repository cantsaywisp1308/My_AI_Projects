#Inspired by PyImageSearch
import numpy as np
import cv2
import argparse


image_path = 'pics/room.jpeg'
prototxt_path = 'models/MobileNetSSD_deploy.prototxt'
model_path = 'models/MobileNetSSD_deploy.caffemodel'
min_confidence = 0.2

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

np.random.seed(543210)
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)


# image= cv2.imread(image_path)
# height, width = image.shape[0], image.shape[1]
# blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
#detected_objects = net.forward()

image = cv2.imread(image_path)
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843,
	(300, 300), 127.5)

print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

# for i in range(detected_objects.shape[2]):
#
#     confidence = detected_objects[0][0][i][2]
#
#     if confidence > min_confidence:
#         class_index = int(detected_objects[0,0,i,1])
#         #coordinates
#         upper_left_x = int(detected_objects[0,0,i,3] * width)
#         upper_left_y = int(detected_objects[0, 0, i, 4] * height)
#         lower_right_x = int(detected_objects[0, 0, i, 5] * width)
#         lower_right_y = int(detected_objects[0, 0, i, 6] * width)
#
#         prediction_text = f"{classes[class_index]} : {confidence:.2f}%"
#         cv2.rectangle(image, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), colors[class_index], 3)
#         cv2.putText(image, prediction_text, (upper_left_x,
#                     upper_left_y -15 if upper_left_y > 30 else upper_left_y + 15),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[class_index], 2)

for i in np.arange(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	confidence = detections[0, 0, i, 2]
	# filter out weak detections by ensuring the `confidence` is
	# greater than the minimum confidence
	if confidence > min_confidence:
		# extract the index of the class label from the `detections`,
		# then compute the (x, y)-coordinates of the bounding box for
		# the object
		idx = int(detections[0, 0, i, 1])
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		# display the prediction
		label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
		print("[INFO] {}".format(label))
		cv2.rectangle(image, (startX, startY), (endX, endY),
			COLORS[idx], 2)
		y = startY - 15 if startY - 15 > 15 else startY + 15
		cv2.putText(image, label, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

cv2.imshow('Detected Objects', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
