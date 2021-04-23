from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import argparse

from dai2_visio_utils import to_planar


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

args = parser.parse_args()

# fullFrameTracking = args.full_frame

# Start defining a pipeline
pipeline = dai.Pipeline()

# colorCam = pipeline.createColorCamera()
# colorCam.setPreviewSize(300, 300)
# colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
# colorCam.setInterleaved(False)
# colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
# colorCam.setFps(40)


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
objectTracker.setTrackerIdAssigmentPolicy(dai.TrackerIdAssigmentPolicy.SMALLEST_ID)

# if fullFrameTracking:
#     colorCam.video.link(objectTracker.inputTrackerFrame)
# else:
detectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)

detectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
detectionNetwork.out.link(objectTracker.inputDetections)
trackerOut = pipeline.createXLinkOut()
trackerOut.setStreamName("tracklets")
objectTracker.out.link(trackerOut.input)


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
        frame = dai.ImgFrame()
        frame.setType(dai.RawImgFrame.Type(8))
        frame.setSequenceNum(seq_num)
        frame.setWidth(IMG_SIZE)
        frame.setHeight(IMG_SIZE)
        frame.setData(to_planar(vidFrame, (IMG_SIZE, IMG_SIZE)))
        device.getInputQueue("video_in").send(frame)
        seq_num += 1

        track = tracklets.get()

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        color = (255, 0, 0)
        frame_arr = frame.getCvFrame()
        trackletsData = track.tracklets

        for t in trackletsData:
            
            if t.status == dai.Tracklet.TrackingStatus.LOST:
                continue

            roi = t.roi.denormalize(frame_arr.shape[1], frame_arr.shape[0])
            x1 = int(roi.topLeft().x)
            y1 = int(roi.topLeft().y)
            x2 = int(roi.bottomRight().x)
            y2 = int(roi.bottomRight().y)

            try:
                label = labelMap[t.label]
            except:
                label = t.label

            statusMap = {dai.Tracklet.TrackingStatus.NEW : "NEW", dai.Tracklet.TrackingStatus.TRACKED : "TRACKED", dai.Tracklet.TrackingStatus.LOST : "LOST"}
            cv2.putText(frame_arr, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame_arr, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame_arr, statusMap[t.status], (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.rectangle(frame_arr, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

        cv2.putText(frame_arr, "NN fps: {:.2f}".format(fps), (2, frame_arr.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)

        cv2.imshow("tracker", frame_arr)

        if cv2.waitKey(1) == ord('q'):
            break