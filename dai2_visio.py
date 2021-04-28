import argparse
import signal

from dai2_visio_modules import Main


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action="store_true", help="Prevent debug output")
    parser.add_argument('-v', '--video', type=str, help="Path to video file to be used for inference (conflicts with -cam)")
    parser.add_argument('-s', '--size', default=512, type=int)
    parser.add_argument('-w', '--width', default=720, type=int, help="Visualization width. Height is calculated automatically from aspect ratio")
    parser.add_argument('-r', '--record', type=str, help="Path to video file to be used for recorded video.")
    args = parser.parse_args()

    # Create the application
    app = Main(args)

    # Register a graceful CTRL+C shutdown
    def signal_handler(sig, frame):
        app.running = False

    signal.signal(signal.SIGINT, signal_handler)

    # Run the application
    app.run()

    # Print latest NN FPS
    print('FPS: ', app.nn_fps)