from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import argparse

from dai2_visio_utils import to_planar

import serial
btSerial = serial.Serial("/dev/ttyAMA0", baudrate=9600, timeout=0.5)
is_rpi = platform.machine().startswith('arm') or platform.machine().startswith('aarch64')


IMG_SIZE = 416


labelMap = [
    "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
    "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"
]

nnPath = './blobs/tiny-yolo-v4_openvino_2021.2_6shave.blob'

parser = argparse.ArgumentParser()
parser.add_argument('video', type=str)
parser.add_argument('-r', '--record', type=str, help="Path to video file to be used for recorded video.")

args = parser.parse_args()

# full frame disabled for the time being. Need to figure this out as it may improve tracking performance.
# fullFrameTracking = args.full_frame

# Start defining a pipeline
pipeline = dai.Pipeline()

# setting node configs
detectionNetwork = pipeline.createYoloDetectionNetwork()
detectionNetwork.setConfidenceThreshold(0.5)
detectionNetwork.setNumClasses(80)
detectionNetwork.setCoordinateSize(4)
detectionNetwork.setAnchors(np.array([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]))
detectionNetwork.setAnchorMasks({"side26": np.array([1, 2, 3]), "side13": np.array([3, 4, 5])})
detectionNetwork.setIouThreshold(0.5)

detectionNetwork.setBlobPath(nnPath)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)

# Link plugins CAM . NN . XLINK
# colorCam.preview.link(detectionNetwork.input)
video_in = pipeline.createXLinkIn()
video_in.setStreamName("video_in")
video_in.out.link(detectionNetwork.input)


objectTracker = pipeline.createObjectTracker()
objectTracker.setDetectionLabelsToTrack([0, 1, 2, 3, 5, 7])  # track vehicle and bicycle
# possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS
objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
# take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
objectTracker.setTrackerIdAssigmentPolicy(dai.TrackerIdAssigmentPolicy.UNIQUE_ID)

# if fullFrameTracking:
#     colorCam.video.link(objectTracker.inputTrackerFrame)
# else:
detectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)

detectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
detectionNetwork.out.link(objectTracker.inputDetections)
trackerOut = pipeline.createXLinkOut()
trackerOut.setStreamName("tracklets")
objectTracker.out.link(trackerOut.input)


# create video recorder
if self.args.record is not None:
    writer = cv2.VideoWriter(self.args.record, cv2.VideoWriter_fourcc(*'MJPG'), 10, (IMG_SIZE,  IMG_SIZE))


# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:

    # Start the pipeline
    device.startPipeline()

    vid_cap = cv2.VideoCapture(args.video)
    framerate = vid_cap.get(cv2.CAP_PROP_FPS)
    tracklets = device.getOutputQueue("tracklets", 4, False)

    startTime = time.monotonic()
    counter = 0
    fps = 0
    frame = None

    seq_num = 0

    while(True):
        # process video frame
        read_correctly, vidFrame = vid_cap.read()
        if not read_correctly:
            break
        imgFrame = dai.ImgFrame()
        imgFrame.setType(dai.RawImgFrame.Type(8))
        imgFrame.setSequenceNum(seq_num)
        imgFrame.setWidth(IMG_SIZE)
        imgFrame.setHeight(IMG_SIZE)
        imgFrame.setData(to_planar(vidFrame, (IMG_SIZE, IMG_SIZE)))
        device.getInputQueue("video_in").send(imgFrame)
        seq_num += 1

        track = tracklets.get()

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        color = (255, 0, 0)
        frame = frame.getCvFrame()
        trackletsData = track.tracklets

        for t in trackletsData:
            
            if t.status == dai.Tracklet.TrackingStatus.LOST:
                continue

            roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
            x1 = int(roi.topLeft().x)
            y1 = int(roi.topLeft().y)
            x2 = int(roi.bottomRight().x)
            y2 = int(roi.bottomRight().y)

            # bluetooth interfacing
            area = (x2 - x1) * (y2 - y1)
            if t.status == dai.Tracklet.TrackingStatus.TRACKED:
                #If the object is still tracked compare with the previous frame and check if is closer
                #if id in previous_frame_dict and (previous_frame_dict[id]['area'] < current_frame_dict[id]['area']) and current_frame_dict[id]['area'] > 450:
                if area > 10000:
                    print("ALERT ID {} IS CLOSE INMINENT IMPACT".format(t.id))
                    btSerial.write("a".encode())
                #elif id in previous_frame_dict and (previous_frame_dict[id]['area'] < current_frame_dict[id]['area']) and current_frame_dict[id]['area'] > 100:
                elif area > 8000:
                    print("Warning ID {} is getting closer".format(t.id))
                    btSerial.write("w".encode())
                else:
                    btSerial.write("s".encode())

            try:
                label = labelMap[t.label]
            except:
                label = t.label

            statusMap = {dai.Tracklet.TrackingStatus.NEW : "NEW", dai.Tracklet.TrackingStatus.TRACKED : "TRACKED", dai.Tracklet.TrackingStatus.LOST : "LOST"}
            cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, statusMap[t.status], (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)

        cv2.imshow("tracker", frame)

        # write record frame
        if self.args.record is not None:
            writer.write(frame)

        if cv2.waitKey(1) == ord('q'):
            # close video recorder
            writer.release()
            vid_cap.release()
            break