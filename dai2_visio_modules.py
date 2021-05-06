import time
import queue
import threading

import numpy as np
import cv2

import depthai
print('depthai module: ', depthai.__file__)

from dai2_visio_utils import cos_dist, batch_cos_dist, frame_norm, to_planar, create_pipeline

from collections import deque

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker


np.random.seed(100)


class Main:

    FRAMERATE = 30.0
    COSINE_THRESHOLD = 0.3
    NMS_MAX_OVERLAP = 0.3
    COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

    track_label = ["person", "bicycle", "car", "motorbike", "bus", "truck"]
    
    def __init__(self, args):

        self.running = True

        print("framerate: ", self.FRAMERATE)

        self.args = args
        self.frame_queue = queue.Queue()
        self.visualization_queue = queue.Queue(maxsize=4)
        self.nn_fps = 0

        if self.args.video is not None:
            self.cap = cv2.VideoCapture(args.video)
            self.FRAMERATE = self.cap.get(cv2.CAP_PROP_FPS)

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.COSINE_THRESHOLD, None)
        self.tracker = Tracker(metric)

        self.pts = [deque(maxlen=30) for _ in range(9999)]

    def is_running(self):
        if self.running:
            if self.args.video is None:
                return True
            else:
                return self.cap.isOpened()
        return False

    def inference_task(self):

        # Queues
        detection_passthrough = self.device.getOutputQueue("detection_passthrough")
        detection_nn = self.device.getOutputQueue("detection_nn")

        # Match up frames and detections
        try:
            prev_passthrough = detection_passthrough.getAll()[0]
            prev_inference = detection_nn.getAll()[0]
        except RuntimeError:
            pass

        fps = 0
        t_fps = time.time()

        while self.running:
            try:
                # Get current detection
                passthrough = detection_passthrough.getAll()[0]
                inference = detection_nn.getAll()[0]

                # Count NN fps
                fps = fps + 1

                # Combine all frames to current inference
                frames = []
                while True:

                    frm = self.frame_queue.get()
                    # get the frames corresponding to inference
                    cv_frame = np.ascontiguousarray(frm.getData().reshape(3, frm.getHeight(), frm.getWidth()).transpose(1, 2, 0))

                    frames.append(cv_frame)

                    # Break out once all frames received for the current inference
                    if frm.getSequenceNum() >= prev_passthrough.getSequenceNum() - 1:
                        break

                infered_frame = frames[0]

                # Send bboxes to be infered upon
                # batch run
                
                boxes = []
                labels = []

                for det in inference.detections:
                    raw_bbox = [det.xmin, det.ymin, det.xmax, det.ymax]
                    bbox = frame_norm(infered_frame, raw_bbox)
                    boxes.append(bbox)
                    labels.append(det.label)
                    det_frame = infered_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    
                    nn_data = depthai.NNData()
                    nn_data.setLayer("data", to_planar(det_frame, (48, 96)))
                    self.device.getInputQueue("reid_in").send(nn_data)

                features = []
                for det in inference.detections:
                    features.append(self.device.getOutputQueue("reid_nn").get().getFirstLayerFp16())

                if len(features) > 0:
                    detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxes, features)]

                    # Run non-maxima suppression.
                    boxes = np.array([d.tlwh for d in detections])
                    scores = np.array([d.confidence for d in detections])
                    indices = preprocessing.non_max_suppression(boxes, self.NMS_MAX_OVERLAP, scores)
                    detections = [detections[i] for i in indices]

                    # Call the tracker
                    self.tracker.predict()
                    self.tracker.update(detections)

                # deepsort visualisation
                i = int(0)
                indexIDs = []
                boxes = []

                for det in detections:
                    
                    bbox = det.to_tlbr()

                    for frame in frames:
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,255,255), 2)

                for track in self.tracker.tracks:
                    
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    
                    indexIDs.append(int(track.track_id))
                    bbox = track.to_tlbr()
                    color = [int(c) for c in self.COLORS[indexIDs[i] % len(self.COLORS)]]

                    for frame in frames:
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 3)
                        cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1] -50)),0, 5e-3 * 150, (color),2)
                    
                    i += 1
                    
                    center = (int(((bbox[0])+(bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
                    
                    self.pts[track.track_id].append(center)
                    thickness = 5

                    # center point
                    for frame in frames:
                        cv2.circle(frame,  (center), 1, color, thickness)

                    # draw motion path
                    for j in range(1, len(self.pts[track.track_id])):
                        if self.pts[track.track_id][j - 1] is None or self.pts[track.track_id][j] is None:
                            continue
                        thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                        for frame in frames:
                            cv2.line(frame, (self.pts[track.track_id][j-1]), (self.pts[track.track_id][j]),(color), thickness)

                # Send of to visualization thread
                for frame in frames:
                    # put nn_fps
                    # if self.args.debug:
                    cv2.putText(frame, 'NN FPS: '+str(self.nn_fps), (5, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 2)

                    if self.visualization_queue.full():
                        self.visualization_queue.get_nowait()
                    self.visualization_queue.put(frame)
            

                # Move current to prev
                prev_passthrough = passthrough
                prev_inference = inference

                if time.time() - t_fps >= 1.0:
                    self.nn_fps = round(fps / (time.time() - t_fps), 2)
                    fps = 0
                    t_fps = time.time()

            except RuntimeError:
                continue

    def input_task(self):

        seq_num = 0

        while self.is_running():

            if self.args.video is None:
                # Send images to next stage
                try:
                    frame = self.device.getOutputQueue('cam_out').get()
                    self.frame_queue.put(frame)
                except RuntimeError:
                    continue

            else:
                # Get frame from video capture
                read_correctly, vid_frame = self.cap.read()
                if not read_correctly:
                    break

                # Send to NN and to inference thread
                frame_nn = depthai.ImgFrame()
                frame_nn.setSequenceNum(seq_num)
                frame_nn.setWidth(self.args.size)
                frame_nn.setHeight(self.args.size)
                frame_nn.setData(to_planar(vid_frame, (self.args.size, self.args.size)))
                self.device.getInputQueue("detection_in").send(frame_nn)
                self.frame_queue.put(frame_nn)                

                seq_num = seq_num + 1

                # Sleep at video framerate
                time.sleep(1.0 / self.FRAMERATE)
            
        # Stop execution after input task doesn't have
        # any extra data anymore
        self.running = False
        self.cap.release()

    def visualization_task(self):

        first = True

        while self.running:

            t1 = time.time()

            # Show frame if available
            if first or not self.visualization_queue.empty():

                frame = self.visualization_queue.get()
                aspect_ratio = frame.shape[1] / frame.shape[0]
                show_frame = cv2.resize(frame, (int(self.args.width),  int(self.args.width / aspect_ratio)))

                if self.args.record is not None:
                    if first:
                        out = cv2.VideoWriter(self.args.record, cv2.VideoWriter_fourcc(*'MJPG'), 10, (int(self.args.width),  int(self.args.width / aspect_ratio)))
                    
                    out.write(show_frame)
                
                cv2.imshow("frame", show_frame)
                first = False

            # sleep if required
            to_sleep_ms = ((1.0 / self.FRAMERATE) - (time.time() - t1)) * 1000
            key = None
            if to_sleep_ms >= 1:
                key = cv2.waitKey(int(to_sleep_ms)) 
            else:
                key = cv2.waitKey(1)

            # Exit
            if key == ord('q'):
                self.running = False
                
                if self.args.record is not None:
                    out.release()
                
                break

        if self.args.record is not None:
            out.release()

    def run(self):

        pipeline = create_pipeline(self.args)

        # Connect to the device
        with depthai.Device(pipeline) as device:
            self.device = device

            print("Starting pipeline...")
            device.startPipeline()            

            threads = [
                threading.Thread(target=self.input_task),
                threading.Thread(target=self.inference_task), 
            ]
            for t in threads:
                t.start()

            # Visualization task should run in 'main' thread
            self.visualization_task()            

        # cleanup
        self.running = False

        for thread in threads:
            thread.join()