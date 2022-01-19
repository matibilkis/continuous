import numpy as np


###https://people.math.sc.edu/Burkardt/cpp_src/stochastic_rk/stochastic_rk.cpp
#Runge-Kutta Algorithm for the Numerical Integration
#of Stochastic Differential Equations
##
def euler_step(x,t,dt,f,g,parameters):
      w = np.random.normal()
      return dt*f(t,x,parameters) + np.sqrt(dt)*w*g(t,x,parameters)


def rk4_step( x, t, dt, f, g,w, parameters = None):
      a21 =   0.66667754298442
      a31 =   0.63493935027993
      a32 =   0.00342761715422
      a41 = - 2.32428921184321
      a42 =   2.69723745129487
      a43 =   0.29093673271592
      a51 =   0.25001351164789
      a52 =   0.67428574806272
      a53 = - 0.00831795169360
      a54 =   0.08401868181222

      q1 = 3.99956364361748
      q2 = 1.64524970733585
      q3 = 1.59330355118722
      q4 = 0.26330006501868

      t1 = t
      x1 = x
      k1 = dt*f( t1, x1, parameters) + w*np.sqrt(q1)*g( t1, x1, parameters)

      t2 = t1 + (a21*dt)
      x2 = x1 + (a21*k1)
      k2 = dt*f(t2, x2, parameters) + w*np.sqrt(q2)*g(t2, x2, parameters)

      t3 = t1 + (a31 + a32)*dt
      x3 = x1 + (a31*k1) + (a32*k2)
      k3 = dt*f(t3,x3,parameters) + w*np.sqrt(q3)*g(t3,x3,parameters)

      t4 = t1 + (a41 + a42 + a43)*dt
      x4 = x1 + (a41 * k1) + (a42 * k2) + (a43 * k3)

      k4 = dt*f(t4,x4,parameters) + w*np.sqrt(q4)*g(t4,x4,parameters)

      xstar = x1 + (a51*k1) + (a52*k2) + (a53*k3) + (a54*k4)

      return xstar



def RK4(x,t,dt, fv,gv, parameters):

      a21 =   0.66667754298442
      a31 =   0.63493935027993
      a32 =   0.00342761715422#D+00
      a41 = - 2.32428921184321#D+00
      a42 =   2.69723745129487#D+00
      a43 =   0.29093673271592#D+00
      a51 =   0.25001351164789#D+00
      a52 =   0.67428574806272#D+00
      a53 = - 0.00831795169360#D+00
      a54 =   0.08401868181222#D+00

      q1 = 3.99956364361748#D+00
      q2 = 1.64524970733585#D+00
      q3 = 1.59330355118722#D+00
      q4 = 0.26330006501868#D+00

      t1 = t
      x1 = x
      k1 = dt*fv( t1, x1, parameters ) + np.sqrt(dt*q1)*gv( t1, x1, parameters)

      t2 = t1 + (a21 * dt)
      x2 = x1 + (a21 * k1)
      k2 = dt * fv( t2, x2, parameters) + np.sqrt(dt*q2)*gv( t2, x2, parameters)

      t3 = t1 + (a31 * dt)  + (a32 * dt)
      x3 = x1 + (a31 * k1) + (a32 * k2)
      k3 = dt * fv( t3 , x3, parameters) + np.sqrt(dt*q3)*gv( t3, x3, parameters)

      t4 = t1 + (a41 * dt)  + (a42 * dt)  + (a43 * dt)
      x4 = x1 + (a41 * k1) + (a42 * k2) + (a43 * k3)
      k4 = dt * fv( t4, x4, parameters) + np.sqrt(dt*q4)* gv( t4, x4, parameters)

      xstar = x1 + (a51 * k1) + (a52 * k2) + (a53 * k3) + (a54 * k4)
      return xstar
