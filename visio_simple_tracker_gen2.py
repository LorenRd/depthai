from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import argparse

from visio_utils_gen2 import to_planar


IMG_SIZE = 300


labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

nnPath = './blobs/mobilenet-ssd_6_shaves.blob'

parser = argparse.ArgumentParser()
parser.add_argument('video', type=str)

args = parser.parse_args()

# fullFrameTracking = args.full_frame

# Start defining a pipeline
pipeline = dai.Pipeline()

# colorCam = pipeline.createColorCamera()
detectionNetwork = pipeline.createMobileNetDetectionNetwork()
objectTracker = pipeline.createObjectTracker()
trackerOut = pipeline.createXLinkOut()

xlinkOut = pipeline.createXLinkOut()

xlinkOut.setStreamName("preview")
trackerOut.setStreamName("tracklets")

# colorCam.setPreviewSize(300, 300)
# colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
# colorCam.setInterleaved(False)
# colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
# colorCam.setFps(40)

# setting node configs
detectionNetwork.setBlobPath(nnPath)
detectionNetwork.setConfidenceThreshold(0.5)
detectionNetwork.input.setBlocking(False)

# Link plugins CAM . NN . XLINK
# colorCam.preview.link(detectionNetwork.input)
video_in = pipeline.createXLinkIn()
video_in.setStreamName("video_in")
video_in.out.link(detectionNetwork.input)

objectTracker.passthroughTrackerFrame.link(xlinkOut.input)


objectTracker.setDetectionLabelsToTrack([2, 6, 7, 14])  # track vehicle and bicycle
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
objectTracker.out.link(trackerOut.input)


# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:

    # Start the pipeline
    device.startPipeline()

    vid_cap = cv2.VideoCapture(args.video)
    framerate = vid_cap.get(cv2.CAP_PROP_FPS)
    preview = device.getOutputQueue("preview", 4, False)
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

        # imgFrame = preview.get()
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