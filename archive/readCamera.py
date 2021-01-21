import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt

pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 15)
profile = pipe.start(config)

# Get depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

clipping_distance_in_meters = .9 #1.2 meter
clipping_distance = clipping_distance_in_meters / depth_scale #Converts meters to the pixel values...

cal_clipping_distance = 5.0 / depth_scale

### FIGURE OUT HOW THEY GET XYZ VALUES

align_to = rs.stream.color
align = rs.align(align_to)

groundMask = cv2.imread("upperImage.tif", -1)

try:
    while True:
        frames = pipe.wait_for_frames()
        aligned_frames = align.process(frames)
        
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        masked_depth = np.where(depth_image <= groundMask, depth_image, 0)
        color_image = np.asanyarray(color_frame.get_data())

        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(masked_depth, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((color_image, depth_colormap))
        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        elif key & 0xFF == ord('c'):
            # Lets generate a ground remover.... 
            rowIndex = np.arange(np.size(depth_image, axis=0))
            # Create depth image with background clipped and remove negative values
            #bgDepthRemoved = depth_image
            bgDepthRemoved = np.where((depth_image > clipping_distance) | (depth_image <= 0), 0, depth_image)
            # Get ROW average
            rowAverages = np.true_divide(bgDepthRemoved.sum(1), (bgDepthRemoved!=0).sum(1))
            # Count nonzero values for each row (this helps get rid of noise)
            rowElements = np.count_nonzero(bgDepthRemoved,axis=1)

            #Loop through rows to generate 2d line
            relRows = []
            relRowVals = []
            for row in rowIndex:
                if rowAverages[row] > 0 and rowElements[row] > 100:
                    relRows.append(rowIndex[row])
                    relRowVals.append(rowAverages[row])
            # Fit vals to line
            # Please note... this only works if camera is mounted parrallel to ground
            c = np.polyfit(relRows, relRowVals,1)
            fittedRowAverages = np.polyval(c, relRows)

            print("PolyFit Vals: ", c)
            print("Clipping Distance", clipping_distance)
            
            # Lets make a ground calibration mask
            # give it some buffer...
            upperBound = -900
            percentSlope = .95

            rows = np.arange(0, bgDepthRemoved.shape[0])
            lastDepthVal = np.polyval(c, bgDepthRemoved.shape[0])

            # Upper buffer...
            cupper = [c[0]*(1+percentSlope), 0]
            cupper[1] = c[0] * bgDepthRemoved.shape[0] + c[1] + upperBound - cupper[0] * bgDepthRemoved.shape[0]

            upperDepths = np.polyval(cupper, rows)
            upperDepths[upperDepths >= cal_clipping_distance] = cal_clipping_distance
            upperDepths[upperDepths <= 0] = 0

            plt.plot(relRows, relRowVals, relRows, fittedRowAverages, rows, upperDepths)
            plt.ylabel("Average Row Vals")
            plt.xlabel("Row Numbers")

            plt.show()

            # Actually go and make the images
            upperImage = np.ones(bgDepthRemoved.shape, dtype=np.uint16)
            for i in rows:
                upperImage[i,:] = upperImage[i,:] * upperDepths[i]

            cv2.imwrite("upperImage.tif", upperImage)
            groundMask = upperImage



finally:
    pipe.stop()