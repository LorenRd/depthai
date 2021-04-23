import os
from pathlib import Path

import numpy as np
import cv2

import depthai
print('depthai module: ', depthai.__file__)


MODEL_DIR = './blobs'
DETECTION_MODEL = 'person-vehicle-bike-detection-crossroad-1016_8shave'
# DETECTION_MODEL = 'tiny-yolo-v4_openvino_2021.2_6shave'
EMBEDDING_MODEL = 'person-reidentification-retail-0031_openvino_2020.1_4shave'


def cos_dist(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def batch_cos_dist(a, b):
    return np.dot(a, b.T) / np.dot(np.linalg.norm(a, axis=1)[:, None], np.linalg.norm(b, axis=1)[:, None].T)


def frame_norm(frame, bbox):
    return (np.clip(np.array(bbox), 0, 1) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]).astype(int)


def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    resized = cv2.resize(arr, shape)
    return resized.transpose(2, 0, 1)


def create_pipeline(args):
    
    print("Creating pipeline...")
    pipeline = depthai.Pipeline()
    pipeline.setOpenVINOVersion(version=depthai.OpenVINO.Version.VERSION_2020_2)

    if args.video is None:
        # ColorCamera
        print("Creating Color Camera...")
        cam = pipeline.createColorCamera()
        cam.setPreviewSize(args.size, args.size)
        cam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setBoardSocket(depthai.CameraBoardSocket.RGB)
        cam_xout = pipeline.createXLinkOut()
        cam_xout.setStreamName("cam_out")
        cam.preview.link(cam_xout.input)

    # NeuralNetwork
    print("Creating Detection Neural Network...")
    detection_nn = pipeline.createMobileNetDetectionNetwork()
    detection_nn.setBlobPath(os.path.join(MODEL_DIR, DETECTION_MODEL + '.blob'))
    # Confidence
    detection_nn.setConfidenceThreshold(0.5)
    # Increase threads for detection
    detection_nn.setNumInferenceThreads(2)
    # Specify that network takes latest arriving frame in non-blocking manner
    detection_nn.input.setQueueSize(1)
    detection_nn.input.setBlocking(False)

    # detection_nn = pipeline.createYoloDetectionNetwork()
    # detection_nn.setConfidenceThreshold(0.5)
    # detection_nn.setNumClasses(80)
    # detection_nn.setCoordinateSize(4)
    # detection_nn.setAnchors(np.array([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]))
    # detection_nn.setAnchorMasks({"side26": np.array([1, 2, 3]), "side13": np.array([3, 4, 5])})
    # detection_nn.setIouThreshold(0.5)
    # detection_nn.setBlobPath(os.path.join(MODEL_DIR, DETECTION_MODEL + '.blob'))
    # detection_nn.setNumInferenceThreads(2)
    # detection_nn.input.setQueueSize(1)
    # detection_nn.input.setBlocking(False)

    detection_nn_passthrough = pipeline.createXLinkOut()
    detection_nn_passthrough.setStreamName("detection_passthrough")
    detection_nn_passthrough.setMetadataOnly(True)

    if args.video is None:
        print('linked cam.preview to detection_nn.input')
        cam.preview.link(detection_nn.input)
    else:
        detection_in = pipeline.createXLinkIn()
        detection_in.setStreamName("detection_in")
        detection_in.out.link(detection_nn.input)

    detection_nn_xout = pipeline.createXLinkOut()
    detection_nn_xout.setStreamName("detection_nn")
    detection_nn.out.link(detection_nn_xout.input)
    detection_nn.passthrough.link(detection_nn_passthrough.input)

    # NeuralNetwork
    print("Creating Embedding Neural Network...")
    reid_in = pipeline.createXLinkIn()
    reid_in.setStreamName("reid_in")
    reid_nn = pipeline.createNeuralNetwork()
    reid_nn.setBlobPath(os.path.join(MODEL_DIR, EMBEDDING_MODEL + '.blob'))
    
    # Decrease threads for reidentification
    reid_nn.setNumInferenceThreads(2)
    
    reid_nn_xout = pipeline.createXLinkOut()
    reid_nn_xout.setStreamName("reid_nn")
    reid_in.out.link(reid_nn.input)
    reid_nn.out.link(reid_nn_xout.input)

    print("Pipeline created.")
    return pipeline