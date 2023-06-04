import os
import zmq
import json
import time
import datetime
import argparse
import numpy as np
import cv2
from picamera import PiCamera
import paho.mqtt.client as mqtt

import common

debug_show_images_and_predictions = False # Display images and predictions for each inference.

parser = argparse.ArgumentParser(prog="gui.py", description="Battery Spotter GUI")
parser.add_argument("--inference_server_address", type=str, required=True)
parser.add_argument("--inference_server_port", type=int, required=True)
parser.add_argument("--mqtt_broker_address", type=str, required=True)
parser.add_argument("--mqtt_broker_port", type=int, required=True)
parser.add_argument("--client_name", type=str, required=True)
parser.add_argument("--confidence_threshold", type=float, default=0.42)
args = parser.parse_args()

inference_server = (args.inference_server_address, args.inference_server_port)
mqtt_broker = (args.mqtt_broker_address, args.mqtt_broker_port)

mqtt_client = mqtt.Client("Battery Spotter")
mqtt_client.connect(mqtt_broker[0], mqtt_broker[1])

context = zmq.Context()

socket = context.socket(zmq.REQ)
socket.connect(f"tcp://{inference_server[0]}:{inference_server[1]}")

if not os.path.exists("output"):
	os.mkdir("output")

camera_width, camera_height = 640, 480

with PiCamera() as camera:
	camera.resolution = (camera_width, camera_height)
	camera.framerate = 12
	image = np.empty((camera_height, camera_width, 3), dtype=np.uint8)

	request = 0

	while True:
		camera.capture(image, 'rgb')
		input_data = image

		start_time = time.time()

		common.send_array(socket, input_data)

		message = common.recv_array(socket)

		stop_time = time.time()
		print(f"elapsed time: {stop_time - start_time:.3f}s")

		output_data = message

		predictions = []

		for i in range(10):
			if output_data[0,i,5] >= args.confidence_threshold:
				box = output_data[0, i, 1:5]
				print(box)
				box = np.array([box[1], box[0], box[3], box[2]])
				print(f"score: {output_data[0,i,5]:.2f}; class: {int(output_data[0,i,6])}; {box[0:4].astype(int)}")
				rec = np.array([box[0], box[1], box[2]-box[0], box[3]-box[1]], dtype=np.int)
				image = cv2.rectangle(image, rec=rec, color=(255, 0, 0), thickness=2)
				predictions.append(output_data[0,i])

		now = datetime.datetime.now()

		if len(predictions):
			status = mqtt_client.publish("DETECTION", common.encode_data(args.client_name, now, image, np.hstack(predictions)))
			print(f"DETECTION: publish status: {status}")

		status = mqtt_client.publish("IMAGE", common.encode_data(args.client_name, now, image, np.ones((1,), dtype=np.float32)))
		print(f"IMAGE: publish status: {status}")

		if debug_show_images_and_predictions:
			image2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
			image2 = cv2.resize(image2, (image2.shape[1] // 2, image2.shape[0] // 2), interpolation=cv2.INTER_AREA)

			cv2.imshow("Client", image2)
			cv2.waitKey(1)

		request += 1
