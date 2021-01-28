import numpy as np
import cv2
import matplotlib.pyplot as plt

import aprilTagLandmarks as aTL
import cameraDriver as camDriver
import odometry as odometry
import graphSlam as gS
import math
import copy

boolVisualize = True

def main():
    #init objects
    realsenseObject = camDriver.pyrealsense2()
    realsenseObject.start()

    odObject = odometry.odometry()
    odObject.start()

    slamObject = gS.slamSolver(6,50,6,.1,3.0)

    # TODO add other landmarking methdology
    nAprilTags = 6
    dof = 6
    apriltagDetector = aTL.apriltagDetector(realsenseObject.tagIntrinsics)
    oReadings = np.zeros((1,dof))
    tagPositions = np.zeros((1,nAprilTags,dof+1))
    #tagPositions2 = np.zeros((1,dof+1))
    maxReadings = 50

    imuToCamR = np.array([[1,0,0],[0,0,1],[0,-1,0]]) # rotation imu to cam posW = R * poscam
    imuToCamT = np.array([[.01],[.02],[.01]]) # translation imu to cam posW = T + poscam
    newArray = np.zeros((1,4))

    while True:
        image = realsenseObject.color_image
        landmarks = apriltagDetector.detect(image)
        
        # TODO make better IMU position model
        oReadings = np.append(oReadings,np.reshape(odObject.finalOut,(1,6)),axis=0)
        #oReadings = np.append(oReadings,np.zeros((1,dof)), axis=0)
        tagPosition = np.zeros((1,nAprilTags,dof+1))
        #tagPosition2 = np.zeros((1,dof+1))

        for r in landmarks:
            # ignores april tags not on cube
            if r.tag_id >= nAprilTags:
                continue
            # Convert R to roll pitch Yaw # Pulled these from 
            angles = np.zeros((3,1))
            pose_R = np.dot(imuToCamR, np.array(r.pose_R))
            angles[0] = np.arctan2(pose_R[1][0],pose_R[0][0])
            angles[1] = np.arctan2(-pose_R[2][0],
                    math.sqrt(pow(pose_R[2][1],2)+pow(pose_R[2][2],2)))
            angles[2] = np.arctan2(pose_R[2][1], pose_R[2][2])
            
            pose_t = np.add(np.dot(imuToCamR, r.pose_t), imuToCamT)
            #print("ANGLES: ", angles)
            #print("Translation: ", pose_t," | ", r.pose_t)
            # TODO Convert the sensor readings from camera -> world coordinates
            tagPosition[0,r.tag_id,0] = 1
            tagPosition[0,r.tag_id,1:4] = np.reshape(np.array(pose_t),-1)
            tagPosition[0,r.tag_id,4:7] = np.reshape(angles*180.0/math.pi, (3,))

            #tagPosition2[0,0:4] = tagPosition[0,r.tag_id,0:4]
            #tagPosition2[0,4:7] = tagPosition[0,r.tag_id,4:7]

            # Pulled from Py-Image search
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
            cv2.putText(image, str(r.tag_id), (ptA[0], ptA[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # append matrices for tag positions 
        tagPositions = np.append(tagPositions, tagPosition, axis=0)
        #tagPositions2 = np.append(tagPositions2,tagPosition2, axis=0)
        

        # show the output image after AprilTag detection 
        cv2.imshow("Image", image)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            odObject.close()
            realsenseObject.close()
            break
        
        # Limit Size of Slam algo.. right now it won't stop growing
        # solve some slam
        # Limit Size of Slam algo.. right now it won't stop growing
        # need at least 2 time steps to slam..
        output = slamObject.slam(oReadings, tagPositions)

        # 0 should be X
        # 1 should be Y
        newArray = np.append(newArray,np.array([[output[-7,0],output[-7,1], oReadings[-1,0],oReadings[-1,1]]]),axis=0)
    
    #np.savetxt('landmarkReadings.csv',tagPositions2, delimiter=",")
    #np.savetxt('oReadings.csv',oReadings, delimiter=",")
    np.savetxt('combinedReadings.csv',newArray,delimiter=",")
    plt.plot(output[-6:,0], output[-6:,1], 'bo',output[:-6,0], output[:-6,1], 'r')
    plt.grid(True)
    plt.xlim((-1.5, 1.5))
    plt.ylim((-1.5, 1.5))
    plt.show()


if __name__ == "__main__":
    main()
