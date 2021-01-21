import numpy as np
import math

timeSteps = 4
landMarks = 2

omega = np.zeros((timeSteps*3+landMarks*3,timeSteps*3+landMarks*3))
ep = np.zeros((timeSteps*3+landMarks*3,1))

pos = [[3,4,0],[2.4,-2.7,47],[3.2,2.5,-42],[4.3,-1.6, -46]]
L = [[[3,-1,90],[1,2,45],[],[]],[[7,1,-45],[5,4,-90],[2,2,-45],[]]]


W_o = .5
W_m = 2.0
W_mr= 2.0

#Initialize Weights

for time_step in range(timeSteps):

    if time_step != timeSteps-1:
        dx = pos[time_step+1][0] # The plus 1 leaves pos 1 for initial condition
        dy = pos[time_step+1][1]
        dalpha = pos[time_step+1][2]

    x0 = (time_step*3)
    x1 = x0+3
    y0 = x0+1
    y1 = y0+3
    alpha0 = y0+1
    alpha1 = alpha0+3

    for landMark in range(landMarks):
        lM = L[landMark][time_step]
        if lM == []:
            continue
        Ldx = lM[0]
        Ldy = lM[1]
        Ldalpha = lM[2]

        Lx = (timeSteps*3+(landMark*3))
        Ly = Lx+1
        Lalpha = Ly+1

        # X
        omega[x0,x0] += W_m #outlink to landmark on diag
        omega[Lx,Lx] += W_m #Inlink to landmark on diag
        omega[x0,Lx] += -W_m #-weight between x0 and landmark
        omega[Lx,x0] += -W_m #symetric

        # Y
        omega[y0,y0] += W_m #outlink to landmark on diag
        omega[Ly,Ly] += W_m #inlink to landmark on diag
        omega[y0,Ly] += -W_m #-weight between y0 and landmark
        omega[Ly,y0] += -W_m #symmetric

        # alpha
        omega[alpha0, alpha0] += W_mr #outlink to landmark on diag
        # Do this if variable does not constrain... (Like you can't 0)
        #omega[Lalpha, Lalpha] = 1 #inlink to landmark on diag
        omega[Lalpha, Lalpha] += 1 #inlink to landmark on diag
        omega[alpha0, Lalpha] += -W_mr #-weight between alpha0 and landmark
        omega[Lalpha, alpha0] += -W_mr #symmetric

        # Update ep
        ep[x0] -= Ldx * W_m #Landmarks are always going to be an outlink to nodes
        ep[Lx] += Ldx * W_m #nodes will always be inlinks to landmarks

        ep[y0] -= Ldy * W_m
        ep[Ly] += Ldy * W_m

        ep[alpha0] -= Ldalpha * W_mr
        ep[Lalpha] += Ldalpha * W_mr

    # Now update the node to node relationships between x0 and x1 (Where x1 is the next point)
    # The last point doesn't have a next point...
    # Initial position...
    # Initial Postion
    if time_step == 0:
        omega[x0,x0] += 1
        omega[y0,y0] += 1
        omega[alpha0,alpha0] += 1

        dx0i = pos[time_step][0] # 0 pos is intial condition
        dy0i = pos[time_step][1]
        dalphai = pos[time_step][2]

        ep[x0] += 1 * dx0i
        ep[y0] += 1 * dy0i
        ep[alpha0] += 1 * dalphai
        
    # Ignore last step change...
    if time_step != timeSteps-1:
        omega[x0,x0] += W_o #Outlink to node x1
        omega[x1,x1] += W_o #Inlink from node x0 ## Won't this update the Landmark diagnols?
        omega[x0,x1] += -W_o #-weight between x0 and x1
        omega[x1,x0] += -W_o #symmetric

        omega[y0,y0] += W_o
        omega[y1,y1] += W_o
        omega[y0,y1] += -W_o
        omega[y1,y0] += -W_o

        omega[alpha0,alpha0] += W_o
        omega[alpha1,alpha1] += W_o
        omega[alpha0,alpha1] += -W_o
        omega[alpha1,alpha0] += -W_o

        # Update ep
        ep[x0] -= dx * W_o
        ep[y0] -= dy * W_o
        ep[alpha0] -= dalpha * W_o

        ep[x1] += dx * W_o
        ep[y1] += dy * W_o
        ep[alpha1] += dalpha * W_o


#print(omega.shape)
#print(np.sum(omega,axis=0))
#print(ep)

X = np.linalg.solve(omega, ep)
print(np.reshape(X, (-1,3)))





