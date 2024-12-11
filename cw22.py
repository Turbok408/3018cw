import numpy as np
import matplotlib.pyplot as plt

def L(t,y,y_dot,alpha,beta):
    """
    :param t: time
    :param y: y value
    :param y_dot:  value of y'
    :param alpha: alpha value
    :param beta: beta value
    :return: value of equation P(t;y')-y at values specified
    """
    return alpha*y_dot**2 + beta*(t**2 - 1)* y_dot **3 - y

def solve_ivp(alpha,beta,y_dot_0,y_0,h):
    """
    :param alpha: alpha value for P
    :param beta: beta value for P
    :param y_dot_0: intial value for y'(0)
    :param y_0: intial value for y(0)
    :param h: step size to use
    :return: (t,y) tuple of t and y values for solved equation 4
    """
    t = np.linspace(0, 1, int(1 / h + 1))
    t= np.concatenate((np.array([0]),t))
    y = np.zeros_like(t)
    y[1] = y_0
    y_dot = np.zeros_like(t)
    y_dot[0] = y_dot_0
    y[0] = y[1] - y_dot[0] * h
    for i in range(1, len(t) - 1):
        d2ld2y_dot = (L(t[i], y[i], y_dot[i] + h, alpha, beta) - 2 * L(t[i], y[i], y_dot[i], alpha, beta) + L(t[i],y[i],y_dot[i] - h,alpha,beta)) / h ** 2
        dLdy = (L(t[i], y[i] + h, y_dot[i], alpha, beta) - L(t[i], y[i] - h, y_dot[i], alpha, beta)) / (2 * h)
        d2dtdy_dot = (L(t[i] + h, y[i], y_dot[i] + h, alpha, beta) - L(t[i] - h, y[i], y_dot[i] + h, alpha, beta) - L(t[i] + h, y[i], y_dot[i] - h, alpha, beta) + L(t[i] - h, y[i], y_dot[i] - h, alpha, beta)) / (4 * h ** 2)
        d2ldydy_dot = (L(t[i], y[i] + h, y_dot[i] + h, alpha, beta) - L(t[i], y[i] - h, y_dot[i] + h, alpha, beta) - L(t[i], y[i] + h, y_dot[i] - h, alpha, beta) + L(t[i], y[i] - h, y_dot[i] - h, alpha, beta)) / (4 * h ** 2)
        y[i + 1] = (dLdy * h ** 2 - d2dtdy_dot * h ** 2 + d2ldydy_dot * h * y[i] - d2ld2y_dot * y[i - 1] + 2 * d2ld2y_dot * y[i]) / (d2ldydy_dot * h + d2ld2y_dot)
        y_dot[i + 1] = (y[i + 1] - y[i]) / h
    return np.delete(t,0),np.delete(y,0)

def shoot(target,error,alpha,beta,h):
    """
    :param target: target value for y(1) to shoot for
    :param error: error to stop at
    :param alpha: alpha value in P
    :param beta: beta value in P
    :param h: h value to solve eq 4 with
    :return: value of y'(0) given that y(1) = target
    """
    # start guess at 10 and decrease guess by a step each time
    guess = 10
    step = 1
    found = False
    while not found:
        result = solve_ivp(alpha,beta,guess,1,h)[1][-1]
        #if result for y(1) with allowed error stop guessing
        if np.isclose(result,target,rtol=error):
            found = True
        # if still havnt overshot target reduced guess and try again
        elif result > target:
            guess -= step
        #if overshot target and not within error go back one guess and reduce step size
        else:
            guess+=step
            step /=10
    return guess

def test_shoot_convergence(target,alpha,beta,h):
    allowed_errors = [1/(10*i) for i in range(6,100)]
    final_y_error = []
    for i in allowed_errors:
        y_dot_intial=shoot(target,i,alpha,beta,h)
        final_y_error.append(np.abs((solve_ivp(alpha, beta, y_dot_intial, 1, h)[1][-1]-target)/target))
    return allowed_errors,final_y_error

for i in range(1,8):
    y_dot_0 = shoot(0.9,0.000000001,5,5,(1/3**i))
    t,y = solve_ivp(5,5,y_dot_0,1,(1/3**i))
    plt.plot(t, y, label="h = "+str((1/4**i)))
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title("Finite Difference Solution For α = 5 = β With Different h Values")
plt.legend()
plt.grid()
plt.show()
y_dot_0 = shoot(0.9,0.000000001,7/4,5,0.15)
t,y = solve_ivp(5, 5, y_dot_0, 1, 0.15)
plt.plot(t, y, label="h = 0.15")
for i in range(2,8):
    y_dot_0 = shoot(0.9,0.000000001,7/4,5,(1/3**i))
    t,y = solve_ivp(5,5,y_dot_0,1,(1/3**i))
    plt.plot(t, y, label="h = "+str((1/3**i)))
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title("Finite Difference Solution For α = 5, β=7/4 With Different h Values")
plt.legend()
plt.grid()
plt.show()
t,y = test_shoot_convergence(0.9,5,5,0.001)
plt.plot(t,y)
plt.xlabel("shooting error")
plt.ylabel("% error for y(1) = 0.9")
plt.title("Error in y(1) Value For Varying Errors For the Shooting Function")
plt.grid()
plt.show()

