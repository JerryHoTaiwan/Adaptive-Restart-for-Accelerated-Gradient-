import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def proj(x, a, b):
    # a < x < b
    x_b = np.minimum(x, b)
    x_a_b = np.maximum(x_b, a)
    return x_a_b

def grad_func(x, P, p):
    P_diag = np.diagonal(P)[:, np.newaxis]
    g1 = np.multiply(P_diag, x)
    g2 = P @ x
    return g1 + g2 + p

def obj_func(x, P, p):
    return x.T @ P @ x + p.T @ x

def gradient_descent(x, maxiter, P, p, a, b, lr):
    loss_list = []
    for i in range(maxiter):
        # gradient descent
        grad = grad_func(x, P, p)
        x = x - lr * grad    
        x = proj(x, a, b)
        loss = obj_func(x, P, p)
        loss_list.append(loss)        
    print ('obj func at iter {}: '.format(i), loss)
    return np.array(loss_list).squeeze()

def nestrov_paper(x, y, maxiter, P, p, a, b, eta):
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

        grad = grad_func(y, P, p)
        x = y - eta * grad
        x = proj(x, a, b)

        #print (x[0], grad[0])

        theta = ((q-theta**2) + np.sqrt((theta**2-q)**2 + 4*theta**2)) / 2
        beta = prev_theta * (1-prev_theta) / (prev_theta**2 + theta)

        y = x + beta * (x - prev_x)
        #x = proj(x, a, b)

        loss = obj_func(x, P, p)

        x1_list.append(x[0])
        x2_list.append(x[1])
        loss_list.append(loss)
    print ('obj func at iter {}: '.format(i), loss)
    return np.array(loss_list).squeeze()

def nestrov_adapt(x, y, maxiter, P, p, a, b, eta):
    loss_list = []
    prev_x = np.copy(x)
    theta = 1
    q = 0
    prev_loss = 10e8
    for i in range(maxiter):
        # nestrov
        prev_x = np.copy(x)
        prev_y = np.copy(y)
        prev_theta = np.copy(theta)

        grad = grad_func(y, P, p)
        x = y - eta * grad
        x = proj(x, a, b)

        if grad.T @ (x - prev_x) > 0:
            print (i)
            theta = 1

        loss = obj_func(x, P, p)
        #if prev_loss < loss:
        #    print (i)
        #    theta = 1
        prev_loss = np.copy(loss)

        theta = ((q-theta**2) + np.sqrt((theta**2-q)**2 + 4*theta**2)) / 2
        beta = prev_theta * (1-prev_theta) / (prev_theta**2 + theta)

        y = x + beta * (x - prev_x)
        #x = proj(x, a, b)

        loss_list.append(loss)
    print ('obj func at iter {}: '.format(i), loss)
    return np.array(loss_list).squeeze()

def nestrov_restart(x, y, maxiter, P, p, a, b, eta):
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

        grad = grad_func(y, P, p)
        x = y - eta * grad
        x = proj(x, a, b)

        theta = ((q-theta**2) + np.sqrt((theta**2-q)**2 + 4*theta**2)) / 2
        beta = prev_theta * (1-prev_theta) / (prev_theta**2 + theta)

        y = x + beta * (x - prev_x)
        #x = proj(x, a, b)
        loss = obj_func(x, P, p)
        loss_list.append(loss)
        #print ('obj func at iter {}: '.format(i), loss)
    print ('obj func at iter {}: '.format(i), loss)
    return np.array(loss_list).squeeze()

if __name__ == '__main__':

    # y = x^T Ax
    # A = [a b]
    #     [c d]
    np.random.seed(42)
    dim = 10
    x = 100 * np.random.normal(0, 1, dim)
    x = x[:, np.newaxis]
    #a = 5 * np.random.normal(0, 1, dim)
    #a = a[:, np.newaxis]
    #b = 5 + 5 * np.random.normal(0, 1, dim)
    #b = b[:, np.newaxis]
    a = (-1) * np.ones((dim, 1))
    b = 1 * np.ones((dim, 1))
    y = np.copy(x)
    maxiter = 500
    lr = 0.001
    #A = np.abs(np.random.normal(0, 1, size=(dim, dim)))
    P = datasets.make_spd_matrix(dim) 
    p = np.random.normal(0, 0.1, dim) 
    p = p[:, np.newaxis]

    w, v = np.linalg.eig(P) 
    lamb = np.amax(np.real(w))
    eta = 1/lamb
    #eta = 0.001

    #f_star = -2.728179
    #f_star = -0.205473
    #f_star = -0.0187818

    loss = obj_func(x, P, p)
    print ('obj func on init: ', loss)

    gd_loss = gradient_descent(x, maxiter, P, p, a, b, lr)
    nd_loss = nestrov_paper(x, y, maxiter, P, p, a, b, eta)
    nd_rs_loss = nestrov_restart(x, y, maxiter, P, p, a, b, eta)
    nd_ad_loss = nestrov_adapt(x, y, maxiter, P, p, a, b, eta)

    f_star = nd_ad_loss - 0.0000001

    iters = np.arange(1, maxiter+1)
    plt.plot(iters, np.log(np.abs((nd_rs_loss-f_star)/f_star)), color='r', label='fixed restart')
    plt.plot(iters, np.log(np.abs((nd_loss-f_star)/f_star)), color='b', label='w/o restart')
    plt.plot(iters, np.log(np.abs((gd_loss-f_star)/f_star)), color='y', label='gradient descent')
    plt.plot(iters, np.log(np.abs((nd_ad_loss-f_star)/f_star)), color='g', label='adaptive restart')
    plt.xlabel('iterations')
    plt.ylabel('log(abs(f-f*/f*))')
    plt.legend()
    plt.show()