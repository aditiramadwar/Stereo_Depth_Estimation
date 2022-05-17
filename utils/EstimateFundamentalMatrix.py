import numpy as np

def norm(pts):
    u = pts[:,0]
    u_mean = np.mean(u, axis=0)
    v = pts[:,1]
    v_mean = np.mean(v, axis=0)
    new_u = u-u_mean
    new_v = v-v_mean
    # # distance
    s = (2/np.mean(new_u**2 + new_v**2))**(0.5)
    T1 = np.diag([s, s, 1])
    T2 = np.array([[1, 0, -u_mean], [0, 1, -v_mean], [0, 0, 1]])
    T = np.dot(T1, T2)
    new_pts = np.column_stack((pts, np.ones(len(pts))))
    x_norm = np.dot(T, new_pts.T).T
    # print(x_norm)
    return x_norm, T

def getFundamentalMatrix(points):
    # calculate fundamental matrix F
    x1 = points[:, 0:2]
    x2 = points[:, 2:4]
    norm1, T1 = norm(x1)
    norm2, T2 = norm(x2)
    A = []
    for i in range(len(norm1)):
        xs, ys, _ = norm1[i] # left
        xp, yp, _ = norm2[i] # right
        A1 = np.array([xs*xp, xp*ys, xp, xs*yp, yp*ys, yp, xs, ys, 1])
        A.append(A1)
    A = np.array(A)

    _, _, VT = np.linalg.svd(A, full_matrices=True)
    F = VT.T[:, -1]
    F = F.reshape(3,3)

    # make F rank 2
    U, S, VT = np.linalg.svd(F)
    S = np.diag(S)
    S[-1,-1] = 0
    F = np.dot(U, np.dot(S, VT))

    #normalize F_orig = t2.T* Fnorm*T1
    F = np.dot(T2.T, np.dot(F, T1))
    return F
    
def getEssensialMatrix(k1, k2, F):
    E = k2.T.dot(F).dot(k1)
    u,s,v = np.linalg.svd(E)
    #rank=2
    s = [1,1,0]
    E = np.dot(u, np.dot(np.diag(s), v))
    return E

