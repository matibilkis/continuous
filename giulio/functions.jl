#using LinearAlgebra
#using SparseArrays
#using BenchmarkTools

function kalman_update(u::Tuple{Vector{Float64},Array{Float64,2}},
        dynamics_parameters::Tuple{Array{Float64,2},Array{Float64,2},Array{Float64,2},Array{Float64,2},Array{Float64,1}},
        dy:: Vector{Float64},
        dt::Float64)

    (x,σ) = u
    (A,C,D,Γ,b)=dynamics_parameters
    tmp =(σ*(C')+Γ')
    #dx = (A*x+b)*dt+tmp*(dy-C*x*dt)
    dx = (A-tmp*C)*x*dt +tmp*dy
    dσ = (A*σ+σ*(A')+D-tmp*(tmp'))*dt
 return (dx,dσ)
end

function signal_matrix(x::Vector{Float64},
                    C::Array{Float64,2},
                    dt::Float64)

    dy =C*(x + pinv(C)*randn(length(x))/sqrt(dt))*dt
  return dy
end

function log_likelihood_update(x::Vector{Float64},
    C::Array{Float64,2},
    dy::Vector{Float64})
    tmp =C*x
    dl = tmp'*(dy-tmp*dt*0.5)
    return dl
end



#to be finished!!!!

function time_trace(N::Int = N,
    dt::Float64 = dt,
    dynamics_parameters::Tuple{Array{Float64,2},Array{Float64,2},Array{Float64,2},Array{Float64,2},Array{Float64,1}} = dynamics_parameters0,
    u::Tuple{Vector{Float64},Array{Float64,2}} = u00)
    (A,C,D,Γ,b) = dynamics_parameters
    Y = Array{Float64}(undef,length(u[1]),N)
    for i = 1:N
        Y[:,i] = signal_matrix(u[1],C,dt)
        u = u .+ kalman_update(u,dynamics_parameters,Y[:,i],dt)

        #print("Processing step $i\u001b[1000D")
    end
    return Y
end






function simulation(
    N::Int,
    u00::Tuple{Vector{Float64},Array{Float64,2}},
    u01::Tuple{Vector{Float64},Array{Float64,2}},
    dt::Float64,
    dynamics_parameters0::Tuple{Array{Float64,2},Array{Float64,2},Array{Float64,2},Array{Float64,2},Array{Float64,1}},
    dynamics_parameters1::Tuple{Array{Float64,2},Array{Float64,2},Array{Float64,2},Array{Float64,2},Array{Float64,1}},
    signal::Array{Float64,2})

    (A1,C1,D1,Γ1,b1)=dynamics_parameters1
    (A0,C0,D0,Γ0,b0)=dynamics_parameters0
    t = 0
    #d = length(u01[1])
    u0 = Array{Tuple{Vector{Float64},Array{Float64,2}}}(undef, N + 1)
    u1 = Array{Tuple{Vector{Float64},Array{Float64,2}}}(undef, N + 1)
    d = length(u00)
    u0[1] = u00
    u1[1] = u01
    logl = Array{Float64}(undef, N + 1)
    logl[1] = 0
    for i = 1:N
        dy=signal[:,i]
        u0[i+1] = u0[i] .+ kalman_update(u0[i],dynamics_parameters0,dy,dt)
        u1[i+1] = u1[i] .+ kalman_update(u1[i],dynamics_parameters1,dy,dt)
        logl[i+1] = log_likelihood_update(u1[i][1],C1,dy)-log_likelihood_update(u0[i][1],C0,dy)
        t += dt
    end
    return (u0, u1, logl, signal ,t);
end





#could this be a valuable option?
#==mutable struct s_state
           x:: Vector{Float64}
           σ:: Array{Float64,2}

       end

fog = s_state([1,1], [1 1;1 1])


mutable struct var
         x
         σ
       end

st =var(1,2)
update!(st)

st
function update!(st)
     st.x = st.x+1;
     st.σ = st.σ+2;
     return nothing
 end

 ==#
