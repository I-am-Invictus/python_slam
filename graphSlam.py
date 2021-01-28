import numpy as np

class slamSolver:
    def __init__(self,dof,maxTimeSteps,maxLandmarks,oWeights, mWeights):
        self.dof = dof
        self.maxTimeSteps = maxTimeSteps
        self.maxLandmarks = maxLandmarks

        # check if weights are specific per dim.. otherwise assign them to all dims
        # W > 1 small error
        # 0 < W < 1 larger error
        if (type(1.0)==type(oWeights) or type(1)==type(oWeights)):
            self.oW = np.empty(self.dof); self.oW.fill(oWeights)
        else:
            self.oW = oWeights

        if (type(1.0)==type(mWeights) or type(1)==type(mWeights)):
            self.mW = np.empty(self.dof); self.mW.fill(mWeights)
        else: 
            self.mW = mWeights

    def slam(self,posMat,landmarksMat):
        # deltasMat if 3 dof (x,y,alpha)
        # should be numpy array..
        # [[x0,y0,alpha0],[x1,y1,alpha1],...[xi,yi,alphai]]
        # Where the [x0, y0, alpha0] is always the starting position
        # Values should be in world coordinates
        # and the last point doesn't go anywhere...
        
        # landmarks mat if 3 dof (x,y,alpha)
        # should be numpy array...
        # row# = timeID
        # col# = landMarkID
        # depth = [bool, Lxi, Lyi, Lalphai] # Bool basically means if there is a connection
        # Values should be in world coordinates
        
        timeSteps = posMat.shape[0]
        landMarks = landmarksMat.shape[1]
        omega = np.zeros((timeSteps*self.dof+landMarks*self.dof,timeSteps*self.dof+landMarks*self.dof))
        xi = np.zeros((timeSteps*self.dof+landMarks*self.dof,1))
        columns = np.zeros((2,self.dof), dtype=int)
        columns[1,:] = np.arange(self.dof)

        for time_step in range(timeSteps):
            # all my columns potion
            columns[0,:] = columns[1,:]
            columns[1,:] = np.add(columns[1,:], self.dof)
            # Row 1 is 0 values
            # Row 2 is 1 values

            # Handle initial Conditions
            if time_step == 0:
                for d in range(self.dof):
                    # Init starting position with 1
                    omega[columns[0,d],columns[0,d]] += 1
                    # Init starting position with link to self
                    xi[columns[0,d]] += 1 * posMat[time_step,d]

            # Handle node to node relations
            if time_step < timeSteps-1:
                for d in range(self.dof):
                    # Outlink to node n+1 from node n
                    omega[columns[0,d],columns[0,d]] += self.oW[d]
                    # Inlink to node n+1 from node n
                    omega[columns[1,d],columns[1,d]] += self.oW[d]
                    # - Weight between node n and node n+1
                    omega[columns[0,d],columns[1,d]] += -self.oW[d]
                    # Makes things symmetric.. Might not need to do this.
                    omega[columns[1,d],columns[0,d]] += -self.oW[d]

                    # Update xi
                    #outlink from node n
                    xi[columns[0,d]] -= posMat[time_step+1,d] * self.oW[d]
                    #inlink to node n+1
                    xi[columns[1,d]] += posMat[time_step+1,d] * self.oW[d]

            lcolumns = np.arange(self.dof, dtype=int) + ((timeSteps-1) * self.dof)
            for li in range(landmarksMat.shape[1]):
                # Check to see if there is actually a connection...
                lcolumns = lcolumns + self.dof
                if landmarksMat[time_step,li,0] == 0 and time_step == timeSteps-1:
                    # I still need to do something here...
                    # Assign the diagnol to 1 if relationship to node point is none
                    for di in range(self.dof):
                        omega[lcolumns[di],lcolumns[di]] = 1
                    continue
                elif landmarksMat[time_step,li,0] == 0:
                    continue

                for di in range(self.dof):
                    # This handles if the sensor can't get rotiation information or something
                    if self.mW[di] == 0: # Check if weight is zero can't get info
                         omega[lcolumns[di],lcolumns[di]] = 1
                    else:
                        # outlink of node n to landmark n
                        omega[columns[0,di],columns[0,di]] += self.mW[di]
                        # inlink of landmark n from node n
                        omega[lcolumns[di],lcolumns[di]] += self.mW[di]
                        # -weight between node n and landmark n
                        omega[columns[0,di],lcolumns[di]] += -self.mW[di]
                        # Make matrix symmetri
                        omega[lcolumns[di],columns[0,di]] += -self.mW[di]

                        # Update xi
                        # nodes will always have landmarks as outlinks
                        xi[columns[0,di]] -= landmarksMat[time_step,li,di+1] * self.mW[di]
                        # landmakrs will ways have nodes as inlinks
                        xi[lcolumns[di]] += landmarksMat[time_step,li,di+1] * self.mW[di]
        # Lets actually slam it...
        X = np.linalg.solve(omega,xi)
        X = np.reshape(X,(-1,self.dof))

        return X

if __name__ == "__main__":
    slamSolverObject = slamSolver(2,50,10,.5,2.0)
    '''
    pos = np.array([[3,4,0],[2.4,-2.7,47],[3.2,2.5,-42],[4.3,-1.6, -46]])
    

    L = np.array([[[1,3,-1,90],[1,7,1,-45]], # timestamp 0, landmark 0 then landmark 1
                [[1,1,2,45],[1,5,4,-90]],
                [[0,0,0,0],[1,2,2,-45]],
                [[0,0,0,0],[0,0,0,0]]])
    '''

    pos = np.array([[3,7],[4.5,-1.5],[2,-3]])
    L = np.array([[[0,0,0]],[[1,-4,-3]],[[1,-6,1]]])

    postionEstimates = slamSolverObject.slam(pos,L)
    print(postionEstimates)








                    
