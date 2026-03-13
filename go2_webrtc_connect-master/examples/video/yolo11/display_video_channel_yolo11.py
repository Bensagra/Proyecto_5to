import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
from time import time

INFER_TIME = 10

# Create an OpenCV window and display a blank image
height, width = 720, 1280  # Adjust the size as needed
img = np.zeros((height, width, 3), dtype=np.uint8)
cv2.imshow('Video', img)
cv2.waitKey(1)  # Ensure the window is created

import asyncio
import logging
import threading
import time
from queue import Queue
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from aiortc import MediaStreamTrack

# Enable logging for debugging
logging.basicConfig(level=logging.FATAL)

def main():
    frame_queue = Queue()

    # Choose a connection method (uncomment the correct one)
    #conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.8.181")
    #conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="10.20.30.30")
    # conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber="B42D2000XXXXXXXX")
    # conn = Go2WebRTCConnection(WebRTCConnectionMethod.Remote, serialNumber="B42D2000XXXXXXXX", username="email@gmail.com", password="pass")
    conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)

    # Load a model
    #model = YOLO("yolo11n.pt")  # load an official model Detection
    model = YOLO("yolo11n-seg.pt")  # load an official model Segmentation
    #model = YOLO("yolo11n-obb.pt")  # load an official model OBB

    # Async function to receive video frames and put them in the queue
    async def recv_camera_stream(track: MediaStreamTrack):
        infer_state = 0
        detections = []
        labels = []
        results = None
        
        while True:
            frame = await track.recv()
            # Convert the frame to RGB format (better acceleration)
            #img_rgb = frame.to_ndarray(format="rgb24")
            #img_bgr = frame.to_ndarray(format="bgr24")
            
            # Used to avoid the FFMPEG warning about converting the pixel format
            # [swscaler @ 0x138018000] No accelerated colorspace conversion found from yuv420p to bgr24.
            yuv = frame.to_ndarray(format="yuv420p")
            img_bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            
            # Convert to PIL Image for inference
            #pil_image = Image.fromarray(img_rgb)
                        
            infer_state += 1
            if infer_state == INFER_TIME:
                # Run inference with PIL Image
                try:
                    # Predict with the model
                    results = model(img_bgr)
                    #results = model(pil_image)  # predict on an image
                    
                    # Access the results
                    #for result in results:
                    #    xywh = result.boxes.xywh  # center-x, center-y, width, height
                    #    xywhn = result.boxes.xywhn  # normalized
                    #    xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
                    #    xyxyn = result.boxes.xyxyn  # normalized
                    #    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
                    #    confs = result.boxes.conf  # confidence score of each box
                        
                except Exception as e:
                    logging.error(f"Yolo11 Predict error: {e}")
                    continue
                
                # visualize the results
                if results and len(results) > 0:
                    result_img = results[0].plot()
                    frame_queue.put(result_img)
                    infer_state = 0
                
            else:
                # visualize the results
                if results and len(results) > 0:
                    result_img = results[0].plot()
                    frame_queue.put(result_img)

    def run_asyncio_loop(loop):
        asyncio.set_event_loop(loop)
        async def setup():
            try:
                # Connect to the device
                await conn.connect()

                # Switch video channel on and start receiving video frames
                conn.video.switchVideoChannel(True)

                # Add callback to handle received video frames
                conn.video.add_track_callback(recv_camera_stream)
            except Exception as e:
                logging.error(f"Error in WebRTC connection: {e}")

        # Run the setup coroutine and then start the event loop
        loop.run_until_complete(setup())
        loop.run_forever()

    # Create a new event loop for the asyncio code
    loop = asyncio.new_event_loop()

    # Start the asyncio event loop in a separate thread
    asyncio_thread = threading.Thread(target=run_asyncio_loop, args=(loop,))
    asyncio_thread.start()

    try:
        while True:
            if not frame_queue.empty():
                img = frame_queue.get()
                #print(f"Shape: {img.shape}, Dimensions: {img.ndim}, Type: {img.dtype}, Size: {img.size}")
                # Display the frame
                cv2.imshow('Video', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # Sleep briefly to prevent high CPU usage
                time.sleep(0.01)
    finally:
        cv2.destroyAllWindows()
        # Stop the asyncio event loop
        loop.call_soon_threadsafe(loop.stop)
        asyncio_thread.join()

if __name__ == "__main__":
    main()
