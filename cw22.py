import numpy as np
import matplotlib.pyplot as plt



def L(t,y,y_dot,alpha,beta):
    return alpha*y_dot**2 + beta*(t**2 - 1)* y_dot **3 - y

def solve_ivp(alpha,beta,y_dot_0,y_0,h):
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
    guess = 10
    step = 1
    found = False
    while not found:
        result = solve_ivp(alpha,beta,guess,1,h)[1][-1]
        if np.isclose(result,target,rtol=error):
            found = True
        elif result > target:
            guess -= step
        else:
            guess+=step
            step /=10
    print((result-target)/target)
    return guess


for i in range(1,8):
    y_dot_0 = shoot(0.9,0.000000001,5,5,(1/3**i))
    t,y = solve_ivp(5,5,y_dot_0,1,(1/3**i))
    plt.plot(t, y, label="h = "+str((1/4**i)))
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title("Finite Difference Solution For α = 5 = β with Different h Values")
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
plt.title("Finite Difference Solution For α = 5, β=7/4 with Different h Values")
plt.legend()
plt.grid()
plt.show()