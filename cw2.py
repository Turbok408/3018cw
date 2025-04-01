import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

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

def solve_ivp(y_dot_0,alpha,beta,y_0,h):
    """
    :param alpha: alpha value for P
    :param beta: beta value for P
    :param y_dot_0: initial value for y'(0)
    :param y_0: initial value for y(0)
    :param h: step size to use
    :return: (t,y) tuple of t and y values for solved equation 4
    """
    #all arrays have duplicate value at beginning as y[-1] value is needed, instead shift arrays by +1 and set y[0]=y[-1]
    #create t array (-h,0....1-h,1)
    t = np.linspace(0, 1, int(1 / h + 1))
    t= np.concatenate((np.array([-h]),t))
    #create y array same length as t y[1]= y_initial
    y = np.zeros_like(t)
    y[1] = y_0
    #create y' array same length as t y[1]= y_initial
    y_dot = np.zeros_like(t)
    y_dot[1] = y_dot_0
    #get y[-1] value from y',y back step
    y[0] = y[1] - y_dot[1] * h
    #as arrays are shifted by 1 start from y[1] using central difference method to find values of the L derivatives
    for i in range(1, len(t) - 1):
        d2ld2y_dot = (L(t[i], y[i], y_dot[i] + h, alpha, beta) - 2 * L(t[i], y[i], y_dot[i], alpha, beta) + L(t[i],y[i],y_dot[i] - h,alpha,beta)) / h ** 2
        dLdy = (L(t[i], y[i] + h, y_dot[i], alpha, beta) - L(t[i], y[i] - h, y_dot[i], alpha, beta)) / (2 * h)
        d2dtdy_dot = (L(t[i] + h, y[i], y_dot[i] + h, alpha, beta) - L(t[i] - h, y[i], y_dot[i] + h, alpha, beta) - L(t[i] + h, y[i], y_dot[i] - h, alpha, beta) + L(t[i] - h, y[i], y_dot[i] - h, alpha, beta)) / (4 * h ** 2)
        d2ldydy_dot = (L(t[i], y[i] + h, y_dot[i] + h, alpha, beta) - L(t[i], y[i] - h, y_dot[i] + h, alpha, beta) - L(t[i], y[i] + h, y_dot[i] - h, alpha, beta) + L(t[i], y[i] - h, y_dot[i] - h, alpha, beta)) / (4 * h ** 2)
        #sub derivative values into eq 4 and solve for y[i+1] using finite difference method
        y[i + 1] = (dLdy * h ** 2 - d2dtdy_dot * h ** 2 + d2ldydy_dot * h * y[i] - d2ld2y_dot * y[i - 1] + 2 * d2ld2y_dot * y[i]) / (d2ldydy_dot * h + d2ld2y_dot)        #find y'[i+1] using finite difference method
        y_dot[i + 1] = (y[i + 1] - y[i]) / h
    # delete t[-1] and y[-1] values
    return np.delete(t,0),np.delete(y,0)

def shoot(target,error,alpha,beta,h):
    """
    :param target: target value for y(1) to shoot for
    :param error: error to stop at
    :param alpha: alpha value in P
    :param beta: beta value in P
    :param h: h value to solve eq 4 with
    :return: value of y'(0) to set y(1) = target
    """
    # choose guess of 10 arbitrarily
    guess = 10
    #use scipy brentq to find a value for y'(0) to make y(1) = target
    y_dot_0 = brentq(lambda x,alpha,beta,y_0,h: solve_ivp(x,alpha,beta,y_0,h)[1][-1]-target,guess,-guess,args=(alpha,beta,1,h),rtol=error)
    #check the algorithm convergers to y(1)=target
    assert np.abs(solve_ivp(y_dot_0,alpha,beta,1,h)[1][-1]-target) < error, "does not converge for h="+str(h)
    return y_dot_0

def test_shoot_convergence(target,alpha,beta,h):
    """
    :param target: target value for y(1)
    :param alpha: alpha value in P
    :param beta: beta value in P
    :param h: h value to solve eq 4 with
    :return: (shooting relative error,error in y(1)=target
    """
    #arbitrary values to get errors for
    allowed_errors = [-0.02*i + 0.5050494949 for i in range(1,25)]
    final_y_error = []
    #for each error find the error in the shooting function returning y'(0) that sets y(1) = 0.9
    for i in allowed_errors:
        y_dot_initial=shoot(target,i,alpha,beta,h)
        final_y_error.append(np.abs((solve_ivp(y_dot_initial,alpha, beta,  1, h)[1][-1]-target)/target))
    return allowed_errors,final_y_error

# solve for α = 5 = β and graph
y_dot_0 = shoot(0.9,0.000000001,5,5,0.0001)
t,y = solve_ivp(y_dot_0,5,5,1,0.0001)
plt.plot(t, y)
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title("Finite Difference Solution For α = 5 = β")
plt.show()
# solve for α = 7/4, β = 5 and graph
y_dot_0 = shoot(0.9,0.00000001,7/4,5,0.0001)
t,y = solve_ivp(y_dot_0,7/4,5,1,0.0001)
plt.plot(t, y)
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title("Finite Difference Solution For α = 7/4, β=5")
plt.show()
# solve for α = 7/4, β = 5 with different h values to show convergence and graph results
h_values = [0.115,0.08,0.04,0.0001]
for i in h_values:
    y_dot_0 = shoot(0.9,0.00000001,7/4,5,i)
    t,y = solve_ivp(y_dot_0,7/4,5,1,i)
    plt.plot(t, y, label="h = "+str(i))
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title("Finite Difference Solution For α = 7/4, β=5 With Different h Values")
plt.legend()
plt.show()
# solve for α = 7/4, β = 5 with different values for the shooting function error to show convergence and graph results
t,y = test_shoot_convergence(0.9,7/4,5,0.0001)
plt.plot(t,y)
plt.xlabel("Shooting relative error")
plt.ylabel("% Error for y(1) = 0.9")
plt.title("Error in y(1) Value For Varying Errors For the Shooting Function,α = 7/4,β=5")
plt.show()

