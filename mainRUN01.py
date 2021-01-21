import numpy as np
import cv2

import aprilTagLandmarks as aTL
import cameraDriver as camDriver
import odometry as odometry
import graphSlam as gS
import math
import copy

def main():
    #init objects
    realsenseObject = camDriver.pyrealsense2()
    realsenseObject.start()


    odObject = odometry.odometry()
    odObject.start()

    nAprilTags = 6
    dof = 6
    apriltagDetector = aTL.apriltagDetector(realsenseObject.tagIntrinsics)
    oReadings = np.zeros((1,dof))
    tagPositions = np.zeros((1,nAprilTags,dof+1))
    maxReadings = 50

    while True:
        image = realsenseObject.color_image
        landmarks = apriltagDetector.detect(image)
        np.append(oReadings,np.reshape(odObject.finalOut,(1,6)),axis=0)
        tagPosition = np.zeros((1,nAprilTags,dof+1))
        for r in landmarks:
            if r.tag_id >= nAprilTags:
                continue
            #print("Tag ID: ", r.tag_id)
            #print("Translation: ", r.pose_t)

            # Convert R to roll pitch Yaw # Pulled these from 
            angles = np.zeros((3,1))
            angles[0] = np.arctan2(r.pose_R[1][0],r.pose_R[0][0])
            angles[1] = np.arctan2(-r.pose_R[2][0],
                    math.sqrt(pow(r.pose_R[2][1],2)+pow(r.pose_R[2][2],2)))
            angles[2] = np.arctan2(r.pose_R[2][1], r.pose_R[2][2])
            tagPosition[0,r.tag_id,0] = 1
            tagPosition[0,r.tag_id,1:4] = np.reshape(np.array(r.pose_t),-1)
            tagPosition[0,r.tag_id,4:7] = np.reshape(angles*180.0/math.pi, (3,))
            #print("Angles: ", angles*180.0/math.pi)
            #print("Odometry: ", np.subtract(newO,lastO))

            # extract the bounding box (x, y)-coordinates for the AprilTag
            # and convert each of the (x, y)-coordinate pairs to integers
            (ptA, ptB, ptC, ptD) = r.corners
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            ptA = (int(ptA[0]), int(ptA[1]))
            # draw the bounding box of the AprilTag detection
            cv2.line(image, ptA, ptB, (0, 255, 0), 2)
            cv2.line(image, ptB, ptC, (0, 255, 0), 2)
            cv2.line(image, ptC, ptD, (0, 255, 0), 2)
            cv2.line(image, ptD, ptA, (0, 255, 0), 2)
            # draw the center (x, y)-coordinates of the AprilTag
            (cX, cY) = (int(r.center[0]), int(r.center[1]))
            cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
            # draw the tag family on the image
            tagFamily = r.tag_family.decode("utf-8")
            cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print("[INFO] tag family: {}".format(tagFamily))
        # show the output image after AprilTag detection 

        np.append(tagPositions, tagPosition)
        cv2.imshow("Image", image)
        key = cv2.waitKey(500)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            odObject.close()
            realsenseObject.close()
            break




if __name__ == "__main__":
    main()
