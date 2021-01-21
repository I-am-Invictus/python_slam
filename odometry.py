import time
from threading import Thread
from mpu9250_jmdev.registers import *
from mpu9250_jmdev.mpu_9250 import MPU9250
import numpy as np
from simple_pid import PID

class odometry(Thread):
    def __init__(self, logData=True):
        Thread.__init__(self)
        # Create
        self.mpu = MPU9250(
            address_ak=AK8963_ADDRESS, 
            address_mpu_master=MPU9050_ADDRESS_68, # In 0x68 Address
            address_mpu_slave=None, 
            bus=1,
            gfs=GFS_1000, 
            afs=AFS_8G, 
            mfs=AK8963_BIT_16, 
            mode=AK8963_MODE_C100HZ)

        self.mpu.configure() # Apply the settings to the registers.
        self.calibrate6DOF() # run initial imu calibration

        self.angle = np.zeros((3,))
        self.position = np.zeros((3,))
        self.velocity = np.zeros((3,))
        self.readings = np.zeros((6,))
        self.finalOut = np.zeros((6,))
        # Handle the threading
        self.daemon = True
        self._is_running = True
        print("[INFO] Finished Initiallzing")
        self.lowPasses = {}
        # TODO: impliment a calibration tool
        a=[1,-1.8592,.8685]
        b=[.0023,.0046,.0023]
        for i in range(6):
            self.lowPasses[i] = lowPass(a,b)

    def close(self):
        print("[INFO] Closing Thread")
        self._is_running = False

    def calibrate6DOF(self):
        self.mpu.calibrate()
        self.mpu.configure()

        self.abias = self.mpu.abias
        self.gbias = self.mpu.gbias

    def run(self):
        ranOnce = 0
        self.endTime = 0.0
        self.startTime = 0.0
        dt = 0.0
        while self._is_running:
            dataFrame = self.mpu.getAllData()
            #['timestamp', 'master_acc_x', 'master_acc_y', 'master_acc_z', 
            # 'master_gyro_x', 'master_gyro_y', 'master_gyro_z', 
            # 'slave_acc_x', 'slave_acc_y', 'slave_acc_z', '
            # slave_gyro_x', 'slave_gyro_y', 'slave_gyro_z', '
            # mag_x', 'mag_y', 'mag_z', 'master_temp', 'slave_temp']
            if ranOnce == 0:
                self.startTime = dataFrame[0]
                ranOnce = 1
                continue
            else:
                # TODO: Clean up method for tracking object motion
                self.endTime = self.startTime
                self.startTime = dataFrame[0]
                dt = self.startTime - self.endTime
                #print(dataFrame)
                dataFrame[3] += -1.0 #Remove gravity in Z... I could use R mat to remove
                    # ax,ay,az is in g
                # gyro is in deg / s
                self.angle += np.array(dataFrame[4:7]) * dt
                self.velocity += np.array(dataFrame[1:4]) * 9.807 * dt
                self.position += self.velocity * dt

                self.readings[3:] = self.angle
                self.readings[0:3] = self.position
                
                # I am putting on the 
                for i in range(6):
                    self.finalOut[i] = self.lowPasses[i].filter(self.readings[i], dt)
                #print("INSIDE O: ", self.finalOut)

class lowPass:
    def __init__(self, a=[1,-1.8592,.8685], b=[.0023, .0046, .0023]):
        self.a = a
        self.b = b
        self.data = np.zeros((3,2))

    def filter(self,newReading,dt):
        self.data[1:] = self.data[0:-1]
        self.data[0,0] = newReading #column 0 is unfliltered data
        self.data[0,1] = (1/self.a[0]) * (self.b[0] * self.data[0,0] + 
                        self.b[1] * self.data[1,0] + self.b[2] * self.data[2,0] -
                        self.a[1] * self.data[1,1] - self.a[2] * self.data[2,1])

        return self.data[0,1]

if __name__=="__main__":
    odObject = odometry()
    odObject.start()
    timeStart = time.time()
    while True:
        time.sleep(1)
        print(odObject.finalOut)
        if time.time() - timeStart > 10.0:
            break
    odObject.close()
