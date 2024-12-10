from operator import truediv

import numpy as np

# Parameters
alpha = 1.0  # Replace with your value
beta = 1.0  # Replace with your value
t0, t_end = 0.0, 1  # Time range
N = 1000  # Number of time steps
dt = (t_end - t0) / N  # Time step size

# Time array and solution initialization
t = np.linspace(t0, t_end, N + 1)
y = np.zeros_like(t)  # Initial guess for y

print(dt)
found = False
target = 5000
step = target/1000
while not found:
    for i in range(1, N+1):
        y[i] = y[i-1]+dt*(3*y[i-1]+2)
    if y[-1] > target:
        found = np.isclose(y[-1], target,rtol=0.000000000001)
        y[0] -=step
        print(y[-1])
        step /= 10
    else:
        y[0]+=step


print(y)
print(step)
# Output the solution
import matplotlib.pyplot as plt

plt.plot(t, y, label="y(t)")
plt.xlabel("Time t")
plt.ylabel("y(t)")
plt.title("Finite Difference Solution")
plt.legend()
plt.grid()
plt.show()
