import numpy as np

n = 16**3
size = 16
A = np.zeros((n,n))
b = np.zeros((n))
for i in range(n):
    ix = int(i % size)
    iy = int(i % (size * size) / size)
    iz = int(i / (size * size))
    if ix > 0:
        A[i,i] += 1
        A[i,i-1] -= 1
        b[i] += 1
    if ix < size - 1:
        A[i,i] += 1
        A[i,i+1] -= 1
        b[i] -= 1
    if iy > 0 :
        A[i,i] += 1
        A[i,i-size] -= 1
        b[i] += 1
    if iy < size - 1:
        A[i,i] += 1
        A[i,i+size] -= 1
        b[i] -= 1
    if iz > 0:
        A[i,i] += 1
        A[i,i-size*size] -= 1
        b[i] += 1
    if iz < size - 1:
        A[i,i] += 1
        A[i,i+size*size] -= 1
        b[i] -= 1




x = np.zeros((n))
r = b - np.dot(A,x)
d = r.copy()
rho = np.dot(np.transpose(r),r)
rho_old = rho
for t in range(100):
    Ad = np.dot(A,d)
    dAd = np.dot(np.transpose(d),Ad)
    alpha = rho / dAd
    x = x + alpha * d
    r = r - alpha * Ad
    rho = np.dot(np.transpose(r),r)
    if(abs(rho) < 1e-10):
        break
    beta = rho / rho_old
    d = r + beta * d
    rho_old = rho

ba = np.dot(A,x)