import numpy as np


###https://people.math.sc.edu/Burkardt/cpp_src/stochastic_rk/stochastic_rk.cpp
#Runge-Kutta Algorithm for the Numerical Integration
#of Stochastic Differential Equations
##
def euler_step(x,t,dt,f,g,parameters):
      w = np.random.normal()
      return dt*f(t,x,parameters) + np.sqrt(dt)*w*g(t,x,parameters)
