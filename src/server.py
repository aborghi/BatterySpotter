import os
import zmq
import json
import time
import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
import sys
import psutil
import argparse

import common

debug_show_images_and_predictions = False # Display images and predictions for each inference.

parser = argparse.ArgumentParser(prog="gui.py", description="Battery Spotter GUI")
parser.add_argument("--mode", default="local", const="local", nargs="?", choices=["local", "ibm_watson_machine_learning"], help="inference mode", required=True)
parser.add_argument("--ibm_api_key", type=str)
parser.add_argument("--ibm_url", type=str)
parser.add_argument("--ibm_space_id", type=str)
parser.add_argument("--ibm_deployment_uid", type=str)
parser.add_argument("--local_model_path", type=str)
parser.add_argument("--confidence_threshold", type=float, default=0.42)
parser.add_argument("--port", type=int, default=5555)
args = parser.parse_args()

use_ibm_watson_machine_learning = args.mode == "ibm_watson_machine_learning"

parser = argparse.ArgumentParser(prog="gui.py", description="Battery Spotter GUI")
parser.add_argument("--mode", default="local", const="local", nargs="?", choices=["local", "ibm_watson_machine_learning"], help="inference mode", required=True)
parser.add_argument("--ibm_api_key", type=str, required=use_ibm_watson_machine_learning)
parser.add_argument("--ibm_url", type=str, required=use_ibm_watson_machine_learning)
parser.add_argument("--ibm_space_id", type=str, required=use_ibm_watson_machine_learning)
parser.add_argument("--ibm_deployment_uid", type=str, required=use_ibm_watson_machine_learning)
parser.add_argument("--local_model_path", type=str, required=not use_ibm_watson_machine_learning)
parser.add_argument("--confidence_threshold", type=float, default=0.42)
parser.add_argument("--port", type=int, default=5555)
args = parser.parse_args()

# Height and width expected by the model.
model_height = 512
model_width = 512

np.set_printoptions(precision=2)

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind(f"tcp://*:{args.port}")

ip_addresses = {k: v[0].address for k,v in psutil.net_if_addrs().items() if k.startswith('en')}
print(f"IP address(es): {ip_addresses}")

print(f"Listening for clients on port {args.port}.")

if use_ibm_watson_machine_learning:
	print(f"Inference engine: TF (remote via IBM Watson Machine Learning).")

	from ibm_watson_machine_learning import APIClient

	wml_credentials = {
	  "apikey": args.ibm_api_key,
	  "url": args.ibm_url
	}

	client = APIClient(wml_credentials)

	client.set.default_space(args.ibm_space_id)
else:
	print(f"Inference engine: TFLite (local).")

	interpreter = tflite.Interpreter(model_path=args.local_model_path, num_threads=8)
	interpreter.allocate_tensors()

	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

print(f"Inference server ready.")

if not os.path.exists("output"):
	os.mkdir("output")

request = 0

while True:
	image = common.recv_array(socket)
	orig_shape = image.shape
	image = cv2.resize(image, (model_height, model_width), interpolation=cv2.INTER_AREA)

	input_data = np.expand_dims(image, 0)

	if use_ibm_watson_machine_learning:
		scoring_payload = {"input_data": [{"values": input_data}]}

		start_time = time.time()
		predictions = client.deployments.score(args.ibm_deployment_uid, scoring_payload)
		stop_time = time.time()
		print(f"inference request: elapsed time: {stop_time - start_time:.3f}s")

		output_data = predictions['predictions'][0]['values']
		output_data = np.array(output_data)[:,:10,:]
	else:
		interpreter.set_tensor(input_details[0]['index'], input_data)

		start_time = time.time()
		interpreter.invoke()
		stop_time = time.time()
		print(f"inference request: elapsed time: {stop_time - start_time:.3f}s")

		output_data = interpreter.get_tensor(output_details[0]['index'])[:,:10,:]

	factor = orig_shape[0] / model_width, orig_shape[1] / model_height
	output_data[...,1:5] *= np.array([factor[0], factor[1], factor[0], factor[1]])

	common.send_array(socket, output_data)

	if debug_show_images_and_predictions:
		image = cv2.resize(image, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_AREA)

		predictions = []

		for i in range(10):
			if output_data[0,i,5] >= args.confidence_threshold:
				box = output_data[0, i, 1:5]
				box = np.array([box[1], box[0], box[3], box[2]])
				print(f"score: {output_data[0,i,5]:.3f}; class: {int(output_data[0,i,6])}; {box[0:4].astype(int)}")
				image = cv2.rectangle(image, box[0:2].astype(int), box[2:4].astype(int), (255, 0, 0), 6)
				image = cv2.putText(image, str(int(output_data[0,i,6])), box[2:4].astype(int), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
				predictions.append(output_data[0,i])

		image = cv2.resize(image, (orig_shape[1] // 2, orig_shape[0] // 2), interpolation=cv2.INTER_AREA)
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

		cv2.imshow("Server", image)
		cv2.waitKey(1)

	request += 1
