#optomech-dynamics in rotating frame under driving and homodyne detection.
function dynamics_matrix(
    u::Array{Float64},
    signal::Array{Float64},
    drive::Array{Float64},
    dynamics_parameters::Array{Float64},
    ) #
    γ, η, Γ, n = dynamics_parameters
    ζ = 4.0 * η * Γ
    #X,Y Vx Vy, C = u;
    #dX = -((0.5 * γ1) * X + 4.0 * η1 * Γ1 * (Vx * X + C * Y) + drive[1]) * dt +sqrt(4.0 * η1 * Γ1) * (Vx * signal[1] + C * signal[2])

    du1 =
        -(0.5 * γ + ζ * u[3]) * u[1] - ζ * u[5] * u[2] +
        drive[1] +
        sqrt(ζ) * (u[3] * signal[1] + u[5] * signal[2])
    du2 =
        -(0.5 * γ + ζ * u[4]) * u[2] - ζ * u[5] * u[1] +
        drive[2] +
        sqrt(ζ) * (u[4] * signal[2] + u[5] * signal[1])
    du3 = γ * (n + 0.5 - u[3]) + Γ - ζ * (u[3] * u[3] + u[5] * u[5]) # σxx
    du4 = γ * (n + 0.5 - u[4]) + Γ - ζ * (u[4] * u[4] + u[5] * u[5]) # σyy
    du5 = -γ * u[5] - ζ * (u[3] + u[4]) * u[5] # σxy
    dl = sqrt(ζ) * (u[1] * signal[1] + u[2] * signal[2]) - 0.5 * ζ *(u[1]^2.0 + u[2]^2.0 ) #loglike
    #dl = sqrt(ζ) * (u[1] * signal[1] ) - 0.5 * ζ *(u[1]^2.0) #loglike

    return [du1, du2, du3, du4, du5, dl]
end

#homodyne current
function signal_matrix(u::Array{Float64}, dt::Float64, dynamics_parameters::Array{Float64})
    γ, η, Γ, n = dynamics_parameters
    ζ = 2.0 * sqrt(η * Γ)
    i1 = ζ * u[1] + randn() / sqrt(dt)
    i2 = ζ * u[2] + randn() / sqrt(dt)
    return [i1, i2] #stochastic current
end


function simulation(
    n::Int,
    u00::Array{Float64},
    u01::Array{Float64},
    dt::Float64,
    dynamics_matrix::Function,
    signal_matrix::Function,
    drive0_matrix::Function,
    drive1_matrix::Function,
    dynamics_parameters0::Array{Float64},
    dynamics_parameters1::Array{Float64},
    #loglike::Function
)
    t = 0
    d = length(u01)
    u1 = Array{Float64}(undef, d, n + 1)
    d = length(u00)
    u0 = Array{Float64}(undef, d, n + 1)
    logl = Array{Float64}(undef, n + 1)
    u0[:, 1] = u00
    u1[:, 1] = u01
    logl[1] = 0
    signaltrace = Array{Float64}(undef, 2, n + 1)
    for i = 1:n
        drive1 = drive1_matrix(t)
        drive0 = drive0_matrix(t)
        signal = signal_matrix(u1[:, i], dt, dynamics_parameters1)
        u0[:, i+1] =
            u0[:, i] + dt * dynamics_matrix(u0[:, i], signal, drive0, dynamics_parameters0)
        u1[:, i+1] =
            u1[:, i] + dt * dynamics_matrix(u1[:, i], signal, drive1, dynamics_parameters1)
        logl[i+1] = (u1[6, i+1] - u0[6, i+1])
        signaltrace[:, i] = signal
        t += dt
    end
    return u0, u1, logl, signaltrace
end




function simulation_sign(
    n::Int,
    u00::Array{Float64},
    u01::Array{Float64},
    dt::Float64,
    dynamics_matrix::Function,
    signal_matrix::Function,
    drive0_matrix::Function,
    drive1_matrix::Function,
    dynamics_parameters0::Array{Float64},
    dynamics_parameters1::Array{Float64},
    #loglike::Function
)
    t = 0
    d = length(u01)
    u1 = Array{Float64}(undef, d, n + 1)
    d = length(u00)
    u0 = Array{Float64}(undef, d, n + 1)
    logl = Array{Float64}(undef, n + 1)
    u0[:, 1] = u00
    u1[:, 1] = u01
    logl[1] = 0
    signaltrace = Array{Float64}(undef, 2, n + 1)
    for i = 1:n

        signal = signal_matrix(u1[:, i], dt, dynamics_parameters1)
        drive1 = signal
        drive0 = signal
        u0[:, i+1] =
            u0[:, i] + dt * dynamics_matrix(u0[:, i], signal, drive0, dynamics_parameters0)
        u1[:, i+1] =
            u1[:, i] + dt * dynamics_matrix(u1[:, i], signal, drive1, dynamics_parameters1)
        logl[i+1] = (u1[6, i+1] - u0[6, i+1])
        signaltrace[:, i] = signal
        t += dt
    end
    return u0, u1, logl, signaltrace
end




#I need to add the average!


#Average function
function avli(
    N::Int,
    n::Int,
    u00::Array{Float64},
    u01::Array{Float64},
    dt::Float64,
    dynamics_matrix::Function,
    signal_matrix::Function,
    drive0_matrix::Function,
    drive1_matrix::Function,
    dynamics_parameters0::Array{Float64},
    dynamics_parameters1::Array{Float64},
)
    u0, u1, logl0, signaltrace = simulation(
        n,
        u00,
        u01,
        dt,
        dynamics_matrix,
        signal_matrix,
        drive0_matrix,
        drive1_matrix,
        dynamics_parameters0,
        dynamics_parameters1
        #loglike::Function
    )
    ts = 0.0:dt:tf
    #must be modified from here!!!
    c = plot(ts, logl0[:])
    Zm = zeros(4, n+1)
    Zm[1, :] += logl[:]
    Zm[2, :] += logl[:].^ 2
    Zm[3, :] += u0[1,:].^2
    Zm[4, :] += u0[1,:]

    for i = 1:N-1
    print("Processing step $i\u001b[1000D")
    u0, u1, logl, signaltrace = simulation(
        n,
        u00,
        u01,
        dt,
        dynamics_matrix,
        signal_matrix,
        drive0_matrix,
        drive1_matrix,
        dynamics_parameters0,
        dynamics_parameters1,
        #loglike::Function
    )

        Zm[1, :] = Zm[1, :] + (logl[:] - Zm[1, :]) / i
        Zm[2, :] = (logl[:].^2.0 + (i - 1.0) * logl[:]) / i
        Zm[3, :] = Zm[3, :]  +(u0[1,:].^ (2) - Zm[3,:])/i
        Zm[4, :] = Zm[4, :]  + (u0[1,:]- Zm[4,:] ) / i
        #Zm[3, :] = Zm[3, :] + (logl[:] - Zm[3, :]) / i
        #solm[i, :] = a[7, :]
        if i < 10
            ts = 0.0:dt:tf
            c = plot!(ts, logl[:])
        end #if
    end #for
    #Zm = Zm / N
    #Zm[2, :] = sqrt.(Zm[2, :] - Zm[1, :] .^ 2)
    #Zm[3, :] = ts[:]
    return Zm, c
end #function








#Average function
function avli2(
    N::Int,
    n::Int,
    u00::Array{Float64},
    u01::Array{Float64},
    dt::Float64,
    dynamics_matrix::Function,
    signal_matrix::Function,
    drive0_matrix::Function,
    drive1_matrix::Function,
    dynamics_parameters0::Array{Float64},
    dynamics_parameters1::Array{Float64},
)
    u0, u1, logl0, signaltrace = simulation2(
        n,
        u00,
        u01,
        dt,
        dynamics_matrix,
        signal_matrix,
        drive0_matrix,
        drive1_matrix,
        dynamics_parameters0,
        dynamics_parameters1
    )
    ts = 0.0:dt:tf
    #must be modified from here!!!
    c = plot(ts, logl0[:])
    Zm = zeros(2, n+1)
    Zm[1, :] += logl[:]
    Zm[2, :] += logl[:].^ 2
    for i = 1:N-1
    u0, u1, logl, signaltrace = simulation2(
            n,
            u00,
            u01,
            dt,
            dynamics_matrix,
            signal_matrix,
            drive0_matrix,
            drive1_matrix,
            dynamics_parameters0,
            dynamics_parameters1
            #loglike::Function
        )
        Zm[1, :] = Zm[1, :] + (logl[:] - Zm[1, :]) / i
        Zm[2, :] = (logl[:] .^ (2) + (i - 1.0) * logl[:]) / i
        #Zm[3, :] = Zm[3, :] + (logl[:] - Zm[3, :]) / i
        #solm[i, :] = a[7, :]
        if i < 10
            ts = 0.0:dt:tf
            c = plot!(ts, logl[:])
        end #if
    end #for
    #Zm = Zm / N
    #Zm[2, :] = sqrt.(Zm[2, :] - Zm[1, :] .^ 2)
    #Zm[3, :] = ts[:]
    return Zm, c
end #function




function simulation3(
    n::Int,
    u00::Array{Float64},
    u01::Array{Float64},
    dt::Float64,
    dynamics_matrix::Function,
    signal_matrix::Function,
    drive0_matrix::Function,
    drive1_matrix::Function,
    dynamics_parameters0::Array{Float64},
    dynamics_parameters1::Array{Float64},
    #loglike::Function
    )
    t = 0
    d = length(u01)
    u1 = Array{Float64}(undef, d, n + 1)
    d = length(u00)
    u0 = Array{Float64}(undef, d, n + 1)
    logl = Array{Float64}(undef, n + 1)
    u0[:, 1] = u00
    u1[:, 1] = u01
    logl[1] = 0
    signaltrace = Array{Float64}(undef, 2, n + 1)
    for i = 1:n
        drive1 = drive1_matrix(t)
        drive0 = drive0_matrix(t)
        signal = signal_matrix(u1[:, i], dt, dynamics_parameters1)
        u0[:, i+1] =
            u0[:, i] + dt * dynamics_matrix(u0[:, i], signal, drive0, dynamics_parameters0)
        if t< 30
        u1[:, i+1] =
            u1[:, i] + dt * dynamics_matrix(u1[:, i], signal, drive1, dynamics_parameters1)
        else
        u1[:, i+1] =
            u1[:, i] + dt * dynamics_matrix(u1[:, i], signal, drive0, dynamics_parameters1)

        end
        logl[i+1] = u1[6, i+1] - u0[6, i+1]
        signaltrace[:, i] = signal
        t += dt
    end
    return u0, u1, logl, signaltrace
end



#In this simulation there is a change in the dynamics at t=change_n.
#The code is simulating 3 processes, maybe this is not really useful and we can reduce it to just compute
function simulation_change_point(
    n::Int,
    u00::Array{Float64},
    u01::Array{Float64},
    dt::Float64,
    dynamics_matrix::Function,
    signal_matrix::Function,
    drive0_matrix::Function,
    drive1_matrix::Function,
    dynamics_parameters0::Array{Float64},
    dynamics_parameters1::Array{Float64},
    #loglike::Function
    change_n::Float64 #time at which the change in the dynamics occurs
    )
    t = 0.
    d = length(u01)
    u1 = Array{Float64}(undef, d, n + 1)
    d = length(u00)
    u0 = Array{Float64}(undef, d, n + 1)
    uexp = Array{Float64}(undef, d, n + 1)
    logl = Array{Float64}(undef, n + 1)
    u0[:, 1] = u00
    u1[:, 1] = u01
    uexp[:,1] = u01
    logl[1] = 0
    tmp = 0.
    signaltrace = Array{Float64}(undef, 2, n + 1)
    for i = 1:n
        drive1 = drive1_matrix(t)
        drive0 = drive0_matrix(t)
        signal = signal_matrix(uexp[:, i], dt, dynamics_parameters1)
        u0[:, i+1] =
            u0[:, i] + dt * dynamics_matrix(u0[:, i], signal, drive0, dynamics_parameters0)
        u1[:, i+1] =
            u1[:, i] + dt * dynamics_matrix(u1[:, i], signal, drive1, dynamics_parameters1)
        if i < change_n/dt
            uexp[:, i+1] =
                uexp[:, i] + dt * dynamics_matrix(uexp[:, i], signal, drive0, dynamics_parameters0)
        else
            uexp[:, i+1] =
                uexp[:, i] + dt * dynamics_matrix(uexp[:, i], signal, drive1, dynamics_parameters1)
        end
        tmp = (u1[6, i+1] - u0[6, i+1])

        (tmp < 0) ? (logl[i+1]= 0) : (logl[i+1]= tmp)

        signaltrace[:, i] = signal
        t += dt
    end
    return u0, u1, logl, signaltrace
end


#in this function each time the log is reset also the state at time t of the  "alternative hypothesis" is reset to be equal to the one of the null hypothesis.
function simulation_change_point_modified(
    n::Int,
    u00::Array{Float64},
    u01::Array{Float64},
    dt::Float64,
    dynamics_matrix::Function,
    signal_matrix::Function,
    drive0_matrix::Function,
    drive1_matrix::Function,
    dynamics_parameters0::Array{Float64},
    dynamics_parameters1::Array{Float64},
    #loglike::Function
    change_n::Float64 #time at which the change in the dynamics occurs
    )
    t = 0.
    d = length(u01)
    u1 = Array{Float64}(undef, d, n + 1)
    d = length(u00)
    u0 = Array{Float64}(undef, d, n + 1)
    uexp = Array{Float64}(undef, d, n + 1)
    logl = Array{Float64}(undef, n + 1)
    u0[:, 1] = u00
    u1[:, 1] = u01
    uexp[:,1] = u01
    logl[1] = 0
    tmp = 0.
    signaltrace = Array{Float64}(undef, 2, n + 1)
    for i = 1:n
        drive1 = drive1_matrix(t)
        drive0 = drive0_matrix(t)
        signal = signal_matrix(uexp[:, i], dt, dynamics_parameters1)
        u0[:, i+1] =
            u0[:, i] + dt * dynamics_matrix(u0[:, i], signal, drive0, dynamics_parameters0)
        u1[:, i+1] =
            u1[:, i] + dt * dynamics_matrix(u1[:, i], signal, drive1, dynamics_parameters1)
        if i < change_n/dt
            uexp[:, i+1] =
                uexp[:, i] + dt * dynamics_matrix(uexp[:, i], signal, drive0, dynamics_parameters0)
        else
            uexp[:, i+1] =
                uexp[:, i] + dt * dynamics_matrix(uexp[:, i], signal, drive1, dynamics_parameters1)
        end
        tmp = (u1[6, i+1] - u0[6, i+1])
        if tmp <0
         logl[i+1]= 0;
         u1[:, i] = u0[:,i];
        else
          logl[i+1]= tmp
        end
        signaltrace[:, i] = signal
        t += dt
    end
    return u0, u1, logl, signaltrace
end






function simulation_n(
    n::Int,
    u00::Array{Float64},
    u01::Array{Float64},
    dt::Float64,
    dynamics_matrix::Function,
    signal_matrix::Function,
    drive0_matrix::Function,
    drive1_matrix::Function,
    dynamics_parameters0::Array{Float64},
    dynamics_parameters1::Array{Float64},
    #loglike::Function
)
    t = 0
    d = length(u01)
    u1 = Array{Float64}(undef, d, n + 1)
    d = length(u00)
    u0 = Array{Float64}(undef, d, n + 1)
    uu = Array{Float64}(undef, d, n + 1)
    logl = Array{Float64}(undef, n + 1)
    u0[:, 1] = u00
    u1[:, 1] = u01
    uu[:, 1] = [1.,1.,1,1,1.0,1.0]
    logl[1] = 0
    signaltrace = Array{Float64}(undef, 2, n + 1)
    for i = 1:n
        drive1 = drive1_matrix(t)
        drive0 = drive0_matrix(t)
        signal = signal_matrix(uu[:, i], dt, dynamics_parameters1)
        uu[:, i+1] =
            u1[:, i] + dt * dynamics_matrix(u1[:, i], signal, drive1, dynamics_parameters1)
        u0[:, i+1] =
            u0[:, i] + dt * dynamics_matrix(u0[:, i], signal, drive0, dynamics_parameters0)
        u1[:, i+1] =
            u1[:, i] + dt * dynamics_matrix(u1[:, i], signal, drive1, dynamics_parameters1)
        logl[i+1] = u1[6, i+1] - u0[6, i+1]
        signaltrace[:, i] = signal
        t += dt
    end
    return u0, u1, uu , logl, signaltrace
end


################################
#must Check if this function properly works
function stopping_time_simulation(
    A::Float64,
    B::Float64,
    n::Int,
    u00::Array{Float64},
    u01::Array{Float64},
    dt::Float64,
    dynamics_matrix::Function,
    signal_matrix::Function,
    drive0_matrix::Function,
    drive1_matrix::Function,
    dynamics_parameters0::Array{Float64},
    dynamics_parameters1::Array{Float64},
    #loglike::Function
)
    n0=0
    t = 0
    d = length(u01)
    u1 = Array{Float64}(undef, d, n + 1)
    d = length(u00)
    u0 = Array{Float64}(undef, d, n + 1)
    uu = Array{Float64}(undef, d, n + 1)
    logl = Array{Float64}(undef, n + 1)
    u0[:, 1] = u00
    u1[:, 1] = u01
    uu[:, 1] = u00 #[1.,1.,1,1,1.0,1.0]
    logl[1] = 0
    signaltrace = Array{Float64}(undef, 2, n + 1)
    for i = 1:n
        drive1 = drive1_matrix(t)
        drive0 = drive0_matrix(t)
        signal = signal_matrix(uu[:, i], dt, dynamics_parameters1)
        uu[:, i+1] =
            u1[:, i] + dt * dynamics_matrix(u1[:, i], signal, drive1, dynamics_parameters1)
        u0[:, i+1] =
            u0[:, i] + dt * dynamics_matrix(u0[:, i], signal, drive0, dynamics_parameters0)
        u1[:, i+1] =
            u1[:, i] + dt * dynamics_matrix(u1[:, i], signal, drive1, dynamics_parameters1)
        logl[i+1] = u1[6, i+1] - u0[6, i+1]
        signaltrace[:, i] = signal
        t += dt
        if  (logl[i] ≥ B || logl[i] ≤A)
            n0=i
            break
        end
    end
    u1_tmp = u1[:,1:n0]
    u0_tmp = u0[:,1:n0]
    uu_tmp = uu[:,1:n0]
    logl_tmp = logl[1:n0]
    signaltrace_tmp = signaltrace[:,1:n0]
    return u0_tmp, u1_tmp ,logl_tmp , signaltrace_tmp
end



function avg_stopping_time(
    A::Float64,
    B::Float64,
    N:: Int,
    n::Int,
    u00::Array{Float64},
    u01::Array{Float64},
    dt::Float64,
    dynamics_matrix::Function,
    signal_matrix::Function,
    drive0_matrix::Function,
    drive1_matrix::Function,
    dynamics_parameters0::Array{Float64},
    dynamics_parameters1::Array{Float64},
    )
    n0=0
    t = 0
    d = length(u01)
    u1 = Array{Float64}(undef, d, n + 1)
    d = length(u00)
    u0 = Array{Float64}(undef, d, n + 1)
    uu = Array{Float64}(undef, d, n + 1)
    logl = Array{Float64}(undef, n + 1)
    u0[:, 1] = u00
    u1[:, 1] = u01
    uu[:, 1] = [1.,1.,1,1,1.0,1.0]
    logl[1] = 0
    a=0.0
    b=0.0
    avg_stop_time=0
    signaltrace = Array{Float64}(undef, 2, n + 1)
    for j= 1:N
        for i = 1:n
            drive1 = drive1_matrix(t)
            drive0 = drive0_matrix(t)
            signal = signal_matrix(uu[:, i], dt, dynamics_parameters1)
            uu[:, i+1] =
                u1[:, i] + dt * dynamics_matrix(u1[:, i], signal, drive1, dynamics_parameters1)
            u0[:, i+1] =
                u0[:, i] + dt * dynamics_matrix(u0[:, i], signal, drive0, dynamics_parameters0)
            u1[:, i+1] =
                u1[:, i] + dt * dynamics_matrix(u1[:, i], signal, drive1, dynamics_parameters1)
            logl[i+1] = u1[6, i+1] - u0[6, i+1]
            signaltrace[:, i] = signal
            t += dt
            if  (logl[i] ≥ B|| logl[i] ≤ A )
                n0=i
                b+=1*(logl[i] ≥ B)
                a+=1*(logl[i] ≤ A)
                break
            #if  (logl[i] ≥ B|| logl[i] ≤ A )
            #    n0=i
            #    b+=1
            #    break
            #end
            #if (logl[i] ≤ A)
            #    n0=i
            #    a+=1
            #    break
            end
        end
    avg_stop_time = avg_stop_time +(n0 - avg_stop_time) / j
    end
   p_a = a/N
   p_b = b/N
   return p_a, p_b, avg_stop_time*dt
end






##############################
#Exact solution?!? just for the case with no driving force
function logl_exact(dynamics_parameters0,dynamics_parameters1, t)

    γ1, η1, Γ1, n1 = dynamics_parameters1

    γ0, η0, Γ0, n0 = dynamics_parameters0


    σu1 = n1 + 1 / 2 + Γ1 / γ1
    σu0 = n0 + 1 / 2 + Γ0 / γ0

    σ1 = (sqrt(1 + 16.0 * η1 * Γ1 * σu1 / γ1) - 1) * γ1 / (8.0 * η1 * Γ1)
    σ0 = (sqrt(1 + 16.0 * η0 * Γ0 * σu0 / γ0) - 1) * γ0 / (8.0 * η0 * Γ0)

    λ = γ0 + 8.0 * η0 * Γ0 * σ0

    a = (4 * η1 * Γ1 * (σ1^2)) / γ1
    b =
        (4 * η0 * Γ0 * σ0^2) * (
            1 +
            ((16.0 * η1 * Γ1 * σ1) / (γ1 + λ)) +
            (64.0 * (η1 * Γ1 * σ1)^(2) / (γ1 * (γ1 + λ)))
        ) / λ
    c =
        8 *
        (σ0 * σ1 * (η0 * Γ0 * η1 * Γ1)^(0.5)) *
        (γ1+ 4.0 * η1 * Γ1 * σ1 ) / ((γ1 + λ)*γ1)


    tot = 4*(η1 * Γ1 * a + η0 * Γ0 * b - 2* sqrt(η1 * Γ1 * η0 * Γ0) * c) * t
    return [a, b, c, tot]
end



#λ = γ0 + 8.0 * η0 * Γ0 * σ0

#a = 4 * η1 * Γ1 * σ1 * σ1 / γ1
#b =
#    (1 * η0 * Γ0 * σ0^2) * (
#        1 +
#        ((16.0 * η1 * Γ1 * σ1) / (γ1 + λ)) +
#        (64.0 * (η1 * Γ1 * σ1)^(2) / (γ1 * (γ1 + λ)))
#    ) / λ
#c =
#    2 *
#    (σ0 * σ1 * (η0 * Γ0 * η1 * Γ1)^(0.5)) *
#    (γ1+ 4.0 * η1 * Γ1 * σ1 ) / ((γ1 + λ)*γ1)



#from new computations on Onyx booox 40
function logl_exact_new(dynamics_parameters0,dynamics_parameters1, t)
    γ1, η1, Γ1, n1 = dynamics_parameters1

    γ0, η0, Γ0, n0 = dynamics_parameters0

    σu1 = n1 + 1 / 2 + Γ1 / γ1
    σu0 = n0 + 1 / 2 + Γ0 / γ0

    σ1 = (sqrt(1 + 16.0 * η1 * Γ1 * σu1 / γ1) - 1) * γ1 / (8.0 * η1 * Γ1)
    σ0 = (sqrt(1 + 16.0 * η0 * Γ0 * σu0 / γ0) - 1) * γ0 / (8.0 * η0 * Γ0)

    β1 = 2*sqrt(η1*Γ1)
    β0 =2*sqrt(η0*Γ0)
    λ  = γ0 + 8.0 * η0 * Γ0 * σ0

    r₁r₁ = (β1*β1*σ1*σ1)/γ1
    r₁r₀ = (1+β1*β1*σ1/γ1)*(2*β0*β1*σ0*σ1)/(γ1+λ)
    r₀r₀ = (1+(2+(2*β1*β1*σ1)/γ1)*(2*β1*β1*σ1)/(γ1+λ))*(β0*β0*σ0*σ0)/λ

    tmp = 2*(β1*β1*r₁r₁ - 2*sqrt(β1*β0)*r₁r₀ + β0*β0*r₀r₀)

    return tmp*t
end
