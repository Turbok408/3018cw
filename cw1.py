import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

def f(t,q,options):
    """
    :param t: time value to be used
    :param q: q vector - the data at time q
    :param options: values of gamma,epsilon and omega in format (gamma,epsilon,omega)
    :return: the value of equation (2) (float)
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
    :return: compute k1,k2,k3 then return value of equation 4d, the value of qn+1 at time tn+dt (float)
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
    :return: use scipy.optimize.fsolve to find values for k1,k2, then return the value of eq (5c),the value of qn+1 at time tn+dt (float)
    """
    def F(K):
        """
        :param K: the vector K in (6) where K=(k1[0],k1[1],k1[0],k1[1])
        :return: the right hand side of the equation (7) as (4,) (k1[0],k1[1],k1[0],k1[1])
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
    else:
        print("Invalid algorithm choice")
    return x,y,t

def calcError(algo,options,numerator):
    """
    :param algo: algorithm to use, RK3 or GRRK3
    :param options: values of gamma,epsilon and omega in format (gamma,epsilon,omega)
    :param numerator : the numerator of the dt intervals specified in the question either 0.1 or 0.05
    :return: [y_errors , dt], the error compared to the exact
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

def plot_dict(data):
    """
    :param data: dict containing title , labels , data[{label,data}] labels and data in data list in same order t as last in list
    :return: plot to screen of given data
    """
    fig = plt.figure(figsize=(16, 8),constrained_layout=True)
    try:
        fig.suptitle(data["title"])
        data["data"][0]["data"][1] # this will raise and error for the try,except if no data is present as the for loop will not
        # for n axis of data create n-1 subplots with n[-1] as its x axis and n as its y axis
        for i in range(len(data["data"][0]["data"]) - 1):
            #create a subplot at with 1 row n-1 columns
            axs = fig.add_subplot(1, len(data["data"][0]["data"])-1, i + 1)
            # for each set of data create a new graph on that same subplot
            for j in data["data"]:
                axs.plot(j["data"][-1], j["data"][i], label=j["label"])
            # if more than one graph on the subplot make a legend
            if len(data["data"]) > 1:
                axs.legend(loc="best")
            axs.set_xlabel(data["labels"][-1])
            axs.set_ylabel(data["labels"][i])
    except KeyError as e:
        print("Missing "+e.args[0]+" key in dict")
    except IndexError:
        print("Less than 2 list of data given")
    except ValueError:
        print("Data sets different length")
    plt.show()

def get_polyfit_data(data,algo):
    """
    :param data: data to fit to polynomial
    :param algo: algorithm  to use as label
    :return: data dict of polynomial fit to data given: {"label":(str),data[y,t]}
    """
    #create an empty t array with min max same as given t values
    t= np.linspace(data[1][0],data[1][-1],1000)
    #fit a 3rd degree polynomial to the data
    z = np.polyfit( data[1],  data[0], 3)
    #create array of values from polynomial
    poly = np.poly1d(z)
    y = poly(t)
    label=str()
    #create str of the polynomial to use as a label
    for index, value in enumerate(poly.coef):
        if index == 0:
            label += str(np.round(value,15))
        else:
            if value > 0:
                label += "+" + str(np.round(value,15)) + "t^" + str(index)
            else:
                label += str(np.round(value,15)) + "t^" + str(index)
    return {"label":label+algo,"data": [y,t]}

questions = []
t_lin_space = np.linspace(0,1,1000)
# define questions with data sets title and labels that are to be plotted
questions.append({
    "title":"Non-stiff case",
    "labels":["x","y","t"],
    "data" :
        [{"label":"RK3","data": solveFunc(0.05,"RK3",(-2,0.05,5))},
        {"label":"GRRK3","data": solveFunc(0.05,"GRRK3",(-2,0.05,5))},
         {"label":"Exact","data":[np.sqrt(1+np.cos(t_lin_space)),np.sqrt(2+np.cos(5*t_lin_space)),t_lin_space]}]

})
questions.append({
    "title":"Convergence rate of GRRK3, RK3 in non stiff case",
    "labels":["Error","Time step"],
    "data" :
        [{"label":"RK3","data":calcError("RK3",(-2,0.05,5),0.01)},
         {"label":"GRRK3","data": calcError("GRRK3",(-2,0.05,5),0.01)},
         get_polyfit_data(calcError("RK3",(-2,0.05,5),0.01),"(RK3)"),
         get_polyfit_data(calcError("GRRK3",(-2,0.05,5),0.01),"(GRRK3)")
         ]

})
questions.append({
    "title":"Stiff case using RK3",
    "labels":["x","y","t"],
    "data" :
        [{"label":"RK3","data":solveFunc(0.001,"RK3",(-2*10**5,0.5,20))}]

})
questions.append({
    "title":"Stiff case using GRRK3",
    "labels":["x","y","t"],
    "data" :
        [{"label":"GRRK3","data":solveFunc(0.005,"GRRK3",(-2*10**5,0.5,20))},
         {"label":"Exact","data":[np.sqrt(1+np.cos(t_lin_space)),np.sqrt(2+np.cos(20*t_lin_space)),t_lin_space]}]
})
questions.append({
    "title":"Convergence rate of GRRK3 in stiff case",
    "labels":["Error","Time step"],
    "data" :
        [{"label":"GRRK3","data": calcError("GRRK3",(-2*10**5,0.5,20),0.05)},
         get_polyfit_data(calcError("GRRK3",(-2*10**5,0.5,20),0.05),"")]
})

#plot all questions
for i in questions:
    plot_dict(i)

