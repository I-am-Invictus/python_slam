from threading import Thread
import pyrealsense2 as rs
import numpy as np
import cv2
import time

class pyrealsense2(Thread):
    def __init__(self):
        Thread.__init__(self)
        # Create camera object
        self.pipe = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 15)
        config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 15)
        self.profile = self.pipe.start(config)
        # Get camera params..
        self.colorIntrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.depthIntrinsics = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        self.tagIntrinsics = [self.colorIntrinsics.fx, self.colorIntrinsics.fy,
                    self.colorIntrinsics.ppx, self.colorIntrinsics.ppy]
        # Get depth scale
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.groundMask = cv2.imread("upperImage.tif", -1)

        self.daemon = True
        self._is_running = True

        print("[INFO] Finished Initiallzing")

    def close(self):
        print("[INFO] Closing Thread")
        self._is_running = False
        self.pipe.stop()

    def run(self):
        while self._is_running:
            frames = self.pipe.wait_for_frames()
            aligned_frames = self.align.process(frames)
             # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            self.depth_image = np.asanyarray(aligned_depth_frame.get_data())
            self.masked_depth = np.where(self.depth_image <= self.groundMask, 
                    self.depth_image, 0)
            self.color_image = np.asanyarray(color_frame.get_data())

if __name__ == "__main__":
    realsenseObject = pyrealsense2()
    realsenseObject.start()
    time.sleep(5)
    while True:
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(realsenseObject.masked_depth,
                     alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((realsenseObject.color_image, depth_colormap))
        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            realsenseObject.close()
            break