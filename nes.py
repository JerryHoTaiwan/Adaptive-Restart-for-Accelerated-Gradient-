import numpy as np
import matplotlib.pyplot as plt

def grad_func(x, a, b, c, d):
    grad = np.array([2*a*x[0] + b*x[1], c*x[0] + 2*d*x[1]])
    return grad

def obj_func(x, a, b, c, d):
    return a*x[0]**2 + (b+c)*x[0]*x[1] + d*x[1]**2

def func_x1x2(x1, x2, a, b, c, d):
    return a*x1**2 + (b+c)*x1*x2 + d*x2**2

def gradient_descent(x, maxiter, a, b, c, d, lr):
    x1_list = []
    x2_list = []
    loss_list = []
    for i in range(maxiter):
        # gradient descent
        grad = grad_func(x, a, b, c ,d)
        x = x - lr * grad    
        loss = obj_func(x, a, b, c, d)
        x1_list.append(x[0])
        x2_list.append(x[1])
        loss_list.append(loss)        
        print ('obj func at iter {}: '.format(i), loss)
    return np.array(x1_list), np.array(x2_list), np.array(loss_list)

def nestrov(x, y, maxiter, a, b, c, d, eta, gamma):
    x1_list = []
    x2_list = []
    loss_list = []
    prev_x = np.copy(x)
    for i in range(maxiter):
        # nestrov
        prev_x = np.copy(x)
        grad = grad_func(y, a, b, c ,d)
        y = x + gamma * (x - prev_x)
        x = y - eta * grad
        loss = obj_func(x, a, b, c, d)

        x1_list.append(x[0])
        x2_list.append(x[1])
        loss_list.append(loss)
        #print ('obj func at iter {}: '.format(i), loss)
        print ('obj func at iter {}: '.format(i), loss)
    return np.array(x1_list), np.array(x2_list), np.array(loss_list)

def nestrov_paper(x, y, maxiter, a, b, c, d, eta, gamma):
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

        grad = grad_func(y, a, b, c ,d)
        x = y - eta * grad

        theta = ((q-theta**2) + np.sqrt((theta**2-q)**2 + 4*theta**2)) / 2
        beta = prev_theta * (1-prev_theta) / (prev_theta**2 + theta)

        y = x + beta * (x - prev_x)

        loss = obj_func(x, a, b, c, d)

        x1_list.append(x[0])
        x2_list.append(x[1])
        loss_list.append(loss)
        #print ('obj func at iter {}: '.format(i), loss)
        print ('obj func at iter {}: '.format(i), loss)
    return np.array(x1_list), np.array(x2_list), np.array(loss_list)

def nestrov_adapt(x, y, maxiter, a, b, c, d, eta, gamma):
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

        grad = grad_func(y, a, b, c ,d)
        x = y - eta * grad

        if grad @ (x - prev_x) > 0:
            theta = 1

        theta = ((q-theta**2) + np.sqrt((theta**2-q)**2 + 4*theta**2)) / 2
        beta = prev_theta * (1-prev_theta) / (prev_theta**2 + theta)

        y = x + beta * (x - prev_x)

        loss = obj_func(x, a, b, c, d)

        x1_list.append(x[0])
        x2_list.append(x[1])
        loss_list.append(loss)
        #print ('obj func at iter {}: '.format(i), loss)
        print ('obj func at iter {}: '.format(i), loss)
    return np.array(x1_list), np.array(x2_list), np.array(loss_list)

def nestrov_restart(x, y, maxiter, a, b, c, d, eta, gamma):
    x1_list = []
    x2_list = []
    loss_list = []
    prev_x = np.copy(x)
    theta = 1
    q = 0
    for i in range(maxiter):

        # restart
        if i % 100 == 0:
            theta = 1

        # nestrov
        prev_x = np.copy(x)
        prev_theta = np.copy(theta)

        grad = grad_func(y, a, b, c ,d)
        x = y - eta * grad

        theta = ((q-theta**2) + np.sqrt((theta**2-q)**2 + 4*theta**2)) / 2
        beta = prev_theta * (1-prev_theta) / (prev_theta**2 + theta)

        y = x + beta * (x - prev_x)

        loss = obj_func(x, a, b, c, d)

        x1_list.append(x[0])
        x2_list.append(x[1])
        loss_list.append(loss)
        #print ('obj func at iter {}: '.format(i), loss)
        print ('obj func at iter {}: '.format(i), loss)
    return np.array(x1_list), np.array(x2_list), np.array(loss_list)


if __name__ == '__main__':

    # y = x^T Ax
    # A = [a b]
    #     [c d]
    np.random.seed(42)
    x = 100 * np.random.normal(0, 1, 2)
    y = np.copy(x)
    maxiter = 300
    lr = 0.001
    a = 25
    b = 12
    c = 10
    d = 9

    eta = 0.001
    gamma = 0.0009

    loss = obj_func(x, a, b, c, d)
    print ('obj func on init: ', loss)

    gd_x1, gd_x2, gd_loss = gradient_descent(x, maxiter, a, b, c, d, lr)
    nd_x1, nd_x2, nd_loss = nestrov_paper(x, y, maxiter, a, b, c, d, eta, gamma)
    nd_rs_x1, nd_rs_x2, nd_rs_loss = nestrov_restart(x, y, maxiter, a, b, c, d, eta, gamma)
    nd_ad_x1, nd_ad_x2, nd_ad_loss = nestrov_adapt(x, y, maxiter, a, b, c, d, eta, gamma)

    x1_grid = np.linspace(-50, 50, 500)
    x2_grid = np.linspace(-50, 50, 500)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    f = func_x1x2(X1, X2, a, b, c, d)

    plt.contour(X1, X2, f, 20, cmap='Purples')
    plt.plot(nd_rs_x1, nd_rs_x2, 'r+', label='fixed restart')
    plt.plot(nd_x1, nd_x2, 'b+', label='w/o restart')
    plt.plot(gd_x1, gd_x2, 'y+', label='gradient descent')
    plt.plot(nd_ad_x1, nd_ad_x2, 'g+', label='adaptive restart')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

    iters = np.arange(1, maxiter+1)
    plt.plot(iters, np.log(nd_rs_loss), color='r', label='fixed restart')
    plt.plot(iters, np.log(nd_loss), color='b', label='w/o restart')
    plt.plot(iters, np.log(gd_loss), color='y', label='gradient descent')
    plt.plot(iters, np.log(nd_ad_loss), color='g', label='adaptive restart')
    plt.xlabel('iterations')
    plt.ylabel('loss (log)')
    plt.legend()
    plt.show()