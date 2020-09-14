#! /usr/bin/env python3

import os
import cv2
import numpy as np

base_dir = os.path.dirname(__file__)
prototxt_path = 'deploy.prototxt'
caffemodel_path = 'weights.caffemodel'

model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

if not os.path.exists('updated_images'):
	print("New directory created")
	os.makedirs('updated_images')

if not os.path.exists('faces'):
	print("New directory created")
	os.makedirs('faces')


cap = cv2.VideoCapture(0)

while(True):
	image = cap.read()[1]
	cv2.imshow('Video', image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

	model.setInput(blob)
	detections = model.forward()

	# Создаём боксы вокруг лиц
	for i in range(0, detections.shape[2]):
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		confidence = detections[0, 0, i, 2]

		# If confidence > 0.5, show box around face
		if (confidence > 0.5):
			cv2.rectangle(image, (startX, startY), (endX, endY), (255, 255, 255), 2)

	# Записываем в файл
	cv2.imshow('Video2', image)
	# cv2.imwrite(base_dir + 'updated_images/' + file, image)
	# print("Image " + file + " converted successfully")

	# Identify each face
	for i in range(0, detections.shape[2]):
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		confidence = detections[0, 0, i, 2]

		# If confidence > 0.5, save it as a separate file
		if (confidence > 0.5):
			frame = image[startY:endY, startX:endX]
			# cv2.imwrite(base_dir + 'faces/' + str(i) + '_' + file, frame)
			cv2.imshow('face', frame)