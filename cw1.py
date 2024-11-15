import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize


def f(t,q,options):
    """
    :param t: time value to be used
    :param q: q vector - the data at time q
    :param options: values of gamma,epsilon and omega in format (gamma,epsilon,omega)
    :return: the value of equation (2)
    """
    #get omega,gamma,epsilon values
    omega= options[2]
    gamma = options[0]
    epsilon = options[1]
    #get x,y values from the q vector
    x=q[0][0]
    y=q[1][0]
    #compute and return the value of equation (2)
    return np.dot(np.array([[gamma,epsilon],[epsilon,1]]),np.array([[(-1+x**2-np.cos(t))/(2*x)],[(-2+y**2-np.cos(omega*t))/(2*y)]]))-np.array([[np.sin(t)/(2*x)],[omega*np.sin(omega*t)/(2*y)]])

def MyRK3_step(f,t,qn,dt,options):
    """
    :param f: equation(2) that is being solved
    :param t: current time and time value for q
    :param qn: previous value of q then q n+1 is to be calculated from
    :param dt: time step defined from the map t
    :param options: values of gamma,epsilon and omega in format (gamma,epsilon,omega)
    :return: compute k1,k2,k3 then return value of equation 4d
    """
    k1 = f(t,qn,options)
    k2 = f(t+1/2*dt,qn+dt/2*k1,options)
    k3 = f(t+dt,qn+dt*(-k1+2*k2),options)
    return qn+dt/6*(k1+4*k2+k3)

def MyGRRK3_step(f, t, qn, dt, options):
    i=0
    def F(K):
        result = K-np.squeeze(np.transpose(f(t+dt/3,qn+dt/12*(5*K),options)))
        return result
    k_intial_guess = f(t+dt/3,qn,options)
    result = scipy.optimize.fsolve(F,k_intial_guess)
    print(result)


def solveFunc(dt,algo,options):
    t = np.linspace(0,1,int(1+1/dt))
    y=[np.sqrt(3)]
    x=[np.sqrt(2)]
    #use t[i+1]?
    if algo == 'RK3':
        for i in range(len(t)-1):
            y.append(MyRK3_step(f,t[i],[[x[i]],[y[i]]],dt,options)[1][0])
            x.append(MyRK3_step(f,t[i],[[x[i]],[y[i]]],dt,options)[0][0])
    if algo == 'GRRK3':
        for i in range(len(t)-1):
            y.append(MyGRRK3_step(f, t[i], [[x[i]], [y[i]]], dt, options)[1][0])
            x.append(MyGRRK3_step(f, t[i], [[x[i]], [y[i]]], dt, options)[0][0])
    return x,y,t

def calcError(algo,options):
    yerrors = []
    dt = [0.1/2**i for i in range(0,8)]
    for i in dt:
        x,y,t = solveFunc(i,algo,options)
        yExact = np.sqrt(2+np.cos(options[2]*t))
        yerrors.append(i*np.sum(np.abs(y-yExact)))
    return yerrors,dt

solveFunc(0.1,"GRRK3",(-2,0.05,5))
"""
yerrors,dts = calcError('RK3',(-2,0.05,5))
plt.plot(dts,yerrors)
plt.show()
x,y,t = solveFunc(0.05,"RK3",(-2,0.05,5))
print(t,x)
plt.plot(t,x)
plt.show()
print(y)
plt.plot(t,y)
plt.show()
"""
