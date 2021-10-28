import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def grad_func(x, A):
    A_diag = np.diagonal(A)[:, np.newaxis]
    g1 = np.multiply(A_diag, x)
    g2 = A.T @ x
    return g1 + g2

def obj_func(x, A):
    return x.T @ A @ x

def gradient_descent(x, maxiter, A, lr):
    loss_list = []
    for i in range(maxiter):
        # gradient descent
        grad = grad_func(x, A)
        x = x - lr * grad    
        loss = obj_func(x, A)
        loss_list.append(loss)        
        #print ('obj func at iter {}: '.format(i), loss)
    return np.array(loss_list).squeeze()

def nestrov(x, y, maxiter, A, eta, gamma):
    x1_list = []
    x2_list = []
    loss_list = []
    prev_x = np.copy(x)
    for i in range(maxiter):
        # nestrov
        prev_x = np.copy(x)
        grad = grad_func(y, A)
        y = x + gamma * (x - prev_x)
        x = y - eta * grad
        loss = obj_func(x, A)

        x1_list.append(x[0])
        x2_list.append(x[1])
        loss_list.append(loss)
        #print ('obj func at iter {}: '.format(i), loss)
    return np.array(loss_list).squeeze()

def nestrov_paper(x, y, maxiter, A, eta, gamma):
    x1_list = []
    x2_list = []
    loss_list = []
    prev_x = np.copy(x)
    theta = 1
    q = 0
    for i in range(maxiter):
        # nestrov
        prev_x = np.copy(x)
        prev_y = np.copy(y)
        prev_theta = np.copy(theta)

        grad = grad_func(y, A)
        x = y - eta * grad

        theta = ((q-theta**2) + np.sqrt((theta**2-q)**2 + 4*theta**2)) / 2
        beta = prev_theta * (1-prev_theta) / (prev_theta**2 + theta)

        y = x + beta * (x - prev_x)

        loss = obj_func(x, A)

        x1_list.append(x[0])
        x2_list.append(x[1])
        loss_list.append(loss)
        print ('obj func at iter {}: '.format(i), loss)
    return np.array(loss_list).squeeze()

def nestrov_adapt(x, y, maxiter, A, eta, gamma):
    x1_list = []
    x2_list = []
    loss_list = []
    prev_x = np.copy(x)
    theta = 1
    q = 0
    for i in range(maxiter):
        # nestrov
        prev_x = np.copy(x)
        prev_y = np.copy(y)
        prev_theta = np.copy(theta)

        grad = grad_func(y, A)
        x = y - eta * grad

        if grad.T @ (x - prev_x) > 0:
            theta = 1

        theta = ((q-theta**2) + np.sqrt((theta**2-q)**2 + 4*theta**2)) / 2
        beta = prev_theta * (1-prev_theta) / (prev_theta**2 + theta)

        y = x + beta * (x - prev_x)

        loss = obj_func(x, A)

        x1_list.append(x[0])
        x2_list.append(x[1])
        loss_list.append(loss)
        print ('obj func at iter {}: '.format(i), loss)
    return np.array(loss_list).squeeze()

def nestrov_restart(x, y, maxiter, A, eta, gamma):
    x1_list = []
    x2_list = []
    loss_list = []
    prev_x = np.copy(x)
    theta = 1
    q = 0
    for i in range(maxiter):

        # restart
        if i % 50 == 0:
            theta = 1

        # nestrov
        prev_x = np.copy(x)
        prev_theta = np.copy(theta)

        grad = grad_func(y, A)
        x = y - eta * grad

        theta = ((q-theta**2) + np.sqrt((theta**2-q)**2 + 4*theta**2)) / 2
        beta = prev_theta * (1-prev_theta) / (prev_theta**2 + theta)

        y = x + beta * (x - prev_x)

        loss = obj_func(x, A)

        x1_list.append(x[0])
        x2_list.append(x[1])
        loss_list.append(loss)
        #print ('obj func at iter {}: '.format(i), loss)
        print ('obj func at iter {}: '.format(i), loss)
    return np.array(loss_list).squeeze()

if __name__ == '__main__':

    # y = x^T Ax
    # A = [a b]
    #     [c d]
    np.random.seed(42)
    dim = 150
    x = 100 * np.random.normal(0, 1, dim)
    x = x[:, np.newaxis]
    y = np.copy(x)
    maxiter = 1500
    lr = 0.001
    #A = np.abs(np.random.normal(0, 1, size=(dim, dim)))
    A = datasets.make_spd_matrix(dim)
    print (A)

    eta = 0.001
    gamma = 0.0009

    loss = obj_func(x, A)
    print ('obj func on init: ', loss)

    gd_loss = gradient_descent(x, maxiter, A, lr)
    nd_loss = nestrov_paper(x, y, maxiter, A, eta, gamma)
    nd_rs_loss = nestrov_restart(x, y, maxiter, A, eta, gamma)
    nd_ad_loss = nestrov_adapt(x, y, maxiter, A, eta, gamma)

    iters = np.arange(1, maxiter+1)
    plt.plot(iters, np.log(nd_rs_loss), color='r', label='fixed restart')
    plt.plot(iters, np.log(nd_loss), color='b', label='w/o restart')
    plt.plot(iters, np.log(gd_loss), color='y', label='gradient descent')
    plt.plot(iters, np.log(nd_ad_loss), color='g', label='adaptive restart')
    plt.xlabel('iterations')
    plt.ylabel('loss (log)')
    plt.legend()
    plt.show()