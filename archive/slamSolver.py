import numpy as np

timeSteps = 3
landMarks = 1

omega = np.zeros((timeSteps*2+landMarks*2,timeSteps*2+landMarks*2))

x0_0 = [3,7]
x1_0 = [4.5, -1.5]
x2_1 = [2, -3]

L0 = [0,-5]
L1 = [-4, -3]
L2 = [-6,1]

W_o = .5
W_m = 2.0

# Build the triag
# Time 0
# X
omega[0,2] = -W_o #x0->x1 weight
omega[0,4] = 0 #x0->x2 weight (no relation)
omega[0,6] = -W_m #x0->Lx weight
# Y
omega[1,3] = -W_o #y0->y1 weight
omega[1,5] = 0 #y0->y2 weight (no relation) 
omega[1,7] = -W_m #y0->Ly weight
# Time 1
# X
omega[2,4] = -W_o #x1->x2 weight
omega[2,6] = -W_m #x1->Lx weight
# Y
omega[3,5] = -W_o #y1->y2 weight
omega[3,7] = -W_m #y1->Ly weight
# Time 2
# X
omega[4,6] = -W_m #x2->Lx weight
# Y
omega[5,7] = -W_m #y2->Ly weight

# Build the diagnol
omega[0,0] = 1 + W_o + W_m #Inlinks + outlinks X0
omega[1,1] = 1 + W_o + W_m #Y0
omega[2,2] = W_o + W_o + W_m #X1
omega[3,3] = W_o + W_o + W_m #Y1
omega[4,4] = W_o + W_m #X2
omega[5,5] = W_o + W_m #Y2
omega[6,6] = W_m + W_m + W_m #Lx
omega[7,7] = W_m + W_m + W_m #Ly

omegaDiag = np.diag(np.diag(omega))
omega = np.subtract(np.add(omega, np.transpose(omega)),omegaDiag)

print(omega)
# Build Epsilon
ep = np.zeros((timeSteps*2+landMarks*2,1))

# Fill in based on weighted inlinks - weighted outlinks
ep[0] = (x0_0[0] * 1) - (L0[0] * W_m + x1_0[0] * W_o) #x0
ep[1] = (x0_0[1] * 1) - (L0[1] * W_m + x1_0[1] * W_o) #y0
ep[2] = (x1_0[0] * W_o) - (L1[0] * W_m + x2_1[0] * W_o) #x1
ep[3] = (x1_0[1] * W_o) - (L1[1] * W_m + x2_1[1] * W_o) #y1
ep[4] = (x2_1[0] * W_o) - (L2[0] * W_m) #x2
ep[5] = (x2_1[1] * W_o) - (L2[1] * W_m) #y2
ep[6] = L0[0] * W_m + L1[0] * W_m + L2[0] * W_m #Lx
ep[7] = L0[1] * W_m + L1[1] * W_m + L2[1] * W_m #Ly

X = np.linalg.solve(omega, ep)
print(np.reshape(X, (-1,2)))




