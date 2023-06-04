import re
import sys
import zmq
import json
import numpy as np

def send_array(socket, array):
    md = dict(
        dtype = str(array.dtype),
        shape = array.shape,
    )
    socket.send_string(str(array.dtype), zmq.SNDMORE)
    socket.send_string(str(array.shape), zmq.SNDMORE)
    return socket.send(array)

def recv_array(socket):
    dtype = socket.recv_string()
    shape = socket.recv_string()
    msg = socket.recv()
    buf = memoryview(msg)
    array = np.frombuffer(buf, dtype=dtype)
    return array.reshape([int(x) for x in shape[1:-1].split(',')])

def encode_data(client_id, timestamp, image, predictions):
    t = np.get_printoptions()['threshold']

    data = {
        "client_id":         str(client_id),
        "timestamp":         str(timestamp),
        "image_shape":       str(image.shape),
        "image_data":        image.tobytes().decode("latin1"),
        "predictions_shape": str(predictions.shape),
        "predictions_data":  predictions.tobytes().decode("latin1")
    }

    return json.dumps(data)

def decode_data(data):
    d = json.loads(data)

    client_id = d["client_id"]
    timestamp = d["timestamp"]

    shape = tuple([int(i) for i in re.findall(r'\d+', d["image_shape"])])
    image = np.frombuffer(d["image_data"].encode("latin1"), dtype=np.uint8).reshape(shape)

    shape = tuple([int(i) for i in re.findall(r'\d+', d["predictions_shape"])])
    predictions = np.frombuffer(d["predictions_data"].encode("latin1"), dtype=np.float32).reshape(shape)

    return client_id, timestamp, image, predictions
