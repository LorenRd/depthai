import argparse
import time
import queue
import signal
import threading

import numpy as np
import cv2

import depthai
print('depthai module: ', depthai.__file__)

from visio_utils_gen2 import *
from visio_modules_gen2 import Main


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