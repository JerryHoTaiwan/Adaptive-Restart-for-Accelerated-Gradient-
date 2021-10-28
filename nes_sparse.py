import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import random as sr

def soft_thd(x, alpha):
    sgn = np.sign(x)
    mag = np.abs(x) - alpha
    mag = np.clip(mag, 0, np.inf)
    return  sgn * mag

def obj_func(x, A, b):
    rho = 1
    return (1/2) * np.square(np.linalg.norm((A @ x) - b, 2)) + rho * np.linalg.norm(x, 1)

def ISTA(x, A, b, maxiter, t):
    loss_list = []
    y = np.copy(x)
    theta = 1
    for i in range(maxiter):
        x = soft_thd(x - t * A.T @ (A @ x - b), t)
        loss = obj_func(x, A, b)
        loss_list.append(loss)        
    print (np.sum(x))
    return np.array(loss_list)

def FISTA(x, A, b, maxiter, t):
    loss_list = []
    y = np.copy(x)
    theta = 1
    for i in range(maxiter):
        prev_x = np.copy(x)
        prev_theta = np.copy(theta)
        x = soft_thd(y - t * A.T @ (A @ y - b), t)
        theta = (1+np.sqrt(1+4*theta**2))/2
        beta = (prev_theta - 1)/theta
        y = x + beta * (x - prev_x)
        prev_theta = np.copy(theta)
        loss = obj_func(x, A, b)
        loss_list.append(loss)        
    print (np.sum(x))
    return np.array(loss_list)

def nestrov_restart(x, A, b, maxiter, t):
    loss_list = []
    y = np.copy(x)
    theta = 1
    for i in range(maxiter):
        prev_x = np.copy(x)
        prev_theta = np.copy(theta)
        if i % 100 == 0:
            theta = 1
        x = soft_thd(y - t * A.T @ (A @ y - b), t)
        theta = (1+np.sqrt(1+4*theta**2))/2
        beta = (prev_theta - 1)/theta
        y = x + beta * (x - prev_x)
        prev_theta = np.copy(theta)
        loss = obj_func(x, A, b)
        loss_list.append(loss)        
    print (np.sum(x))
    return np.array(loss_list)

def nestrov_adapt(x, A, b, maxiter, t):
    loss_list = []
    y = np.copy(x)
    theta = 1
    loss = 10e8
    for i in range(maxiter):
        prev_x = np.copy(x)
        prev_y = np.copy(y)
        prev_theta = np.copy(theta)
        prev_loss = np.copy(loss)
        x = soft_thd(y - t * A.T @ (A @ y - b), t)
        #loss = obj_func(x, A, b)
        #print ((y-x).shape, x.shape, ((y-x) @ (x - prev_x)))

        theta = (1+np.sqrt(1+4*theta**2))/2
        beta = (prev_theta - 1)/theta
        y = x + beta * (x - prev_x)
        prev_theta = np.copy(theta)
        loss = obj_func(x, A, b)

        if (prev_y - x) @ (x - prev_x) > 0:
            theta = 1

        #if loss > prev_loss:
        #    theta = 1
        loss_list.append(loss)   
    print (np.sum(x))
    return np.array(loss_list)

if __name__ == '__main__':

    # y = x^T Ax
    # A = [a b]
    #     [c d]
    np.random.seed(42)
    x = np.random.normal(0, 1, 2000)
    y = np.copy(x)
    maxiter = 800
    A = np.random.normal(0, 1, size=(50, 2000))
    b_sparse = sr(2000, 1, density=0.005).A
    noise = np.random.normal(0, 0.01, 50)
    b = (A @ b_sparse).flatten()
    b = b + noise
    #print (b.shape, noise.shape)
    # = np.random.rand(500)
    w, v = np.linalg.eig(A.T@A)
    lamb = np.amax(np.real(w))
    t = 1/lamb
    loss = obj_func(x, A, b)

    gd_loss = FISTA(x, A, b, maxiter, t)
    nd_loss = ISTA(x, A, b, maxiter, t)
    nd_rs_loss = nestrov_restart(x, A, b, maxiter, t)
    nd_ad_loss = nestrov_adapt(x, A, b, maxiter, t)

    iters = np.arange(1, maxiter+1)
    plt.plot(iters, np.log(nd_rs_loss), color='r', label='fixed restart')
    plt.plot(iters, np.log(nd_loss), color='b', label='ISTA')
    plt.plot(iters, np.log(gd_loss), color='y', label='FISTA')
    plt.plot(iters, np.log(nd_ad_loss), color='g', label='adaptive restart')
    plt.xlabel('iterations')
    plt.ylabel('loss (log)')
    plt.legend()
    plt.show()