cd(dirname(@__FILE__))
#using SparseArrays
using LinearAlgebra
#using Plots
using BenchmarkTools
#using Profile

include("functions.jl")
include("parameters0.jl")

#LinearAlgebra.BLAS.set_num_threads(2)

#plotly()
#gr()

#N=1000
#dt=10^-20
N = 500 #number of element of the ensamble
tf=50 #experimental time
tspan = (0.0, tf)
N = Int(1e5)  #number of time steps
dt = tspan[2] /N

@time signal= time_trace()


@time sim = simulation(
     N,
     u00,
     u01,
     dt,
     dynamics_parameters0,
     dynamics_parameters1,
     signal )


Δ=Int(1e1)

ts= 0.0:Δ*dt:tf


plot(ts,a[1:Δ:end])

plot(ts,sim[4][1,1:Δ:end])

a = Array{Float64}(undef,N+1)
for i=1:N+1
    a[i] = sim[2][i][1][1]
end



for i=1:N
    a[i] = sim[3][i]
end

sim[4][1,:]

x=[1,1]
