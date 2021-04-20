import argparse
import time
import queue
import signal
import threading
from pathlib import Path

import numpy as np
import cv2

import depthai
print('depthai module: ', depthai.__file__)

from visio_utils import *


DEBUG = True
WIDTH = 1280


class Main:

    FRAMERATE = 30.0
    
    def __init__(self):

        self.running = True

        print("framerate: ", self.FRAMERATE)

        self.frame_queue = queue.Queue()
        self.visualization_queue = queue.Queue(maxsize=4)
        self.nn_fps = 0

    def is_running(self): return True

    def inference_task(self):

        # Queues
        detection_passthrough = self.device.getOutputQueue("detection_passthrough")
        detection_nn = self.device.getOutputQueue("detection_nn")

        bboxes = []
        results = {}
        results_path = {}
        next_id = 0

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
                for det in inference.detections:
                    raw_bbox = [det.xmin, det.ymin, det.xmax, det.ymax]
                    bbox = frame_norm(infered_frame, raw_bbox)
                    det_frame = infered_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    nn_data = depthai.NNData()
                    nn_data.setLayer("data", to_planar(det_frame, (48, 96)))
                    self.device.getInputQueue("reid_in").send(nn_data)

                 
                # Retrieve infered bboxes
                for det in inference.detections:

                    raw_bbox = [det.xmin, det.ymin, det.xmax, det.ymax]
                    bbox = frame_norm(infered_frame, raw_bbox)

                    reid_result = self.device.getOutputQueue("reid_nn").get().getFirstLayerFp16()

                    for person_id in results:
                        dist = cos_dist(reid_result, results[person_id])
                        if dist > 0.7:
                            result_id = person_id
                            results[person_id] = reid_result
                            break
                    else:
                        result_id = next_id
                        results[result_id] = reid_result
                        results_path[result_id] = []
                        next_id += 1

                    if DEBUG:
                        for frame in frames:
                            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)
                            x = (bbox[0] + bbox[2]) // 2
                            y = (bbox[1] + bbox[3]) // 2
                            results_path[result_id].append([x, y])
                            cv2.putText(frame, str(result_id), (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255))
                            if len(results_path[result_id]) > 1:
                                cv2.polylines(frame, [np.array(results_path[result_id], dtype=np.int32)], False, (255, 0, 0), 2)
                    else:
                        print(f"Saw id: {result_id}")

                # Send of to visualization thread
                for frame in frames:
                    # put nn_fps
                    if DEBUG:
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

        while self.is_running():
            # Send images to next stage
            try:
                frame = self.device.getOutputQueue('cam_out').get()
                self.frame_queue.put(frame)
            except RuntimeError:
                continue
            
        # Stop execution after input task doesn't have
        # any extra data anymore
        self.running = False

    def visualization_task(self):
        
        first = True
        while self.running:

            t1 = time.time()

            # Show frame if available
            if first or not self.visualization_queue.empty():
                frame = self.visualization_queue.get()
                aspect_ratio = frame.shape[1] / frame.shape[0]
                cv2.imshow("frame", cv2.resize(frame, (int(WIDTH),  int(WIDTH / aspect_ratio))))
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
                break

    def run(self):

        pipeline = create_pipeline()

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


if __name__ == "__main__":

    # Create the application
    app = Main()

    # Register a graceful CTRL+C shutdown
    def signal_handler(sig, frame):
        app.running = False

    signal.signal(signal.SIGINT, signal_handler)

    # Run the application
    app.run()
    # Print latest NN FPS
    print('FPS: ', app.nn_fps)