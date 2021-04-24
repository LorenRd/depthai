import time
import queue
import threading

import numpy as np
import cv2

import depthai
print('depthai module: ', depthai.__file__)

from dai2_visio_utils import cos_dist, batch_cos_dist, frame_norm, to_planar, create_pipeline


class Main:

    FRAMERATE = 30.0

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

        results = None
        results_path = []
        results_last_track = []
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

                # batch run
                objects = []
                for det in inference.detections:
                    raw_bbox = [det.xmin, det.ymin, det.xmax, det.ymax]
                    bbox = frame_norm(infered_frame, raw_bbox)
                    det_frame = infered_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    objects.append(to_planar(det_frame, (48, 96))[None, :])

                if len(objects) > 0:
                    objects = np.concatenate(objects, 0)
                    nn_data = depthai.NNData()
                    nn_data.setLayer("data", objects)
                    self.device.getInputQueue("reid_in").send(nn_data)
                    reid_results = self.device.getOutputQueue("reid_nn").get().getFirstLayerFp16()
                 
                # Retrieve infered bboxes
                for i, det in enumerate(inference.detections):
                    
                    raw_bbox = [det.xmin, det.ymin, det.xmax, det.ymax]
                    bbox = frame_norm(infered_frame, raw_bbox)

                    if results is None:
                        results = reid_results
                    else:
                        dists = batch_cos_dist(reid_results, results)

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

                    # if self.args.debug:
                    for frame in frames:
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)
                        x = (bbox[0] + bbox[2]) // 2
                        y = (bbox[1] + bbox[3]) // 2
                        results_path[result_id].append([x, y])
                        cv2.putText(frame, str(result_id), (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255))
                        if len(results_path[result_id]) > 1:
                            cv2.polylines(frame, [np.array(results_path[result_id], dtype=np.int32)], False, (255, 0, 0), 2)
                    # else:
                    #     print(f"Saw id: {result_id}")

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

    def visualization_task(self):
        
        first = True
        while self.running:

            t1 = time.time()

            # Show frame if available
            if first or not self.visualization_queue.empty():
                frame = self.visualization_queue.get()
                aspect_ratio = frame.shape[1] / frame.shape[0]
                cv2.imshow("frame", cv2.resize(frame, (int(self.args.width),  int(self.args.width / aspect_ratio))))
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