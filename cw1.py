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
    :return: compute k1,k2,k3 then return value of equation 4d, the value of qn+1 at time tn+dt
    """
    k1 = f(t,qn,options)
    k2 = f(t+1/2*dt,qn+dt/2*k1,options)
    k3 = f(t+dt,qn+dt*(-k1+2*k2),options)
    return qn+dt/6*(k1+4*k2+k3)

def MyGRRK3_step(f, t, qn, dt, options):
    """
    :param f: equation(2) that is being solved
    :param t: current time and time value for q
    :param qn: previous value of q then q n+1 is to be calculated from
    :param dt: time step defined from the map t
    :param options: values of gamma,epsilon and omega in format (gamma,epsilon,omega)
    :return: use scipy.optimize.fsolve to find values for k1,k2, then return the value of eq (5c),the value of qn+1 at time tn+dt
    """
    def F(K):
        """
        :param K: the vector K in (6) where K=(k1[0],k1[1],k1[0],k1[1])
        :return: the right hand side of the equation (7)
        """
        #scipy.optimize.fsolve changes the initial value K to a list of length 4 with K=(k1[0],k1[1],k1[0],k1[1]), must put the values k1,k2 into individual matrices of shape(2,1)
        k1=np.array([[K[0]],[K[1]]])
        k2 = np.array([[K[2]], [K[3]]])
        #find the value of the RHS of equation (7)
        result = np.array([[k1],[k2]])-np.array([[f(t+dt/3,qn+dt/12*(5*k1-k2),options)],[f(t+dt,qn+dt/4*(3*k1+k2),options)]])
        #scipy.optimize.fsolve expects a (4,) list back so convert the [(2,1),(2,1)] matrix from eq(7) to a (4,)
        result = np.reshape(result,(4,))
        return result
    k_initial_guess = np.array([f(t+dt/3,qn,options),f(t+dt,qn,options)])
    result = scipy.optimize.fsolve(F,[k_initial_guess])
    #the result scipy.optimize.fsolve gives is (4,) to get k1,k2 split the (4,) in half then reshape k1,k2 to be (2,1)
    result = np.split(result,2)
    k1 = np.reshape(result[0],(2,1))
    k2 = np.reshape(result[1],(2,1))
    return qn + dt/4*(3 * k1 + k2)


def solveFunc(dt,algo,options):
    """
    :param dt: time step to solve for
    :param algo: algorithm to use, RK3 or GRRK3
    :param options: values of gamma,epsilon and omega in format (gamma,epsilon,omega)
    :return: list (x,y,t) where x,y are the values of the function f(eq(2))  at time t calculated with specified algorithm and timestep dt
    """
    #create evenly spaced t array with spacing dt
    t = np.linspace(0,1,int(1+1/dt))
    #create x,y lists with initial values
    y=[np.sqrt(3)]
    x=[np.sqrt(2)]
    #using specified algorithm iterate through t using previous x,y value to find the next
    if algo == "RK3":
        for i in range(len(t)-1):
            result = MyRK3_step(f,t[i],[[x[i]],[y[i]]],dt,options)
            y.append(result[1][0])
            x.append(result[0][0])
    elif algo == "GRRK3":
        for i in range(len(t)-1):
            result = MyGRRK3_step(f, t[i], [[x[i]], [y[i]]], dt, options)
            y.append(result[1][0])
            x.append(result[0][0])
    return x,y,t

def calcError(algo,options,numerator):
    """
    :param algo: algorithm to use, RK3 or GRRK3
    :param options: values of gamma,epsilon and omega in format (gamma,epsilon,omega)
    :param numerator : the numerator of the dt intervals specified in the question either 0.1 or 0.05
    :return: tuple of value of y errors at time interval dt and time intervals
    """
    y_errors = []
    #initialse list of time intervals specified in question
    dt = [numerator/2**i for i in range(0,8)]
    #iterate over the list dt finding the error between the exact and calculated y values
    for i in dt:
        x,y,t = solveFunc(i,algo,options)
        y_exact = np.sqrt(2+np.cos(options[2]*t))
        y_errors.append(i*np.sum(np.abs(y-y_exact)))
    return y_errors,dt

x,y,t = solveFunc(0.001,"GRRK3",(-2*10**-5,0.5,20))
plt.plot(t,x)
plt.show()
plt.plot(t,y)
plt.show()

yerrors,dts = calcError("RK3",(-2,0.05,5),0.01)
plt.plot(dts,yerrors)
yerrors,dts = calcError("GRRK3",(-2,0.05,5),0.01)
plt.plot(dts,yerrors)
plt.show()
