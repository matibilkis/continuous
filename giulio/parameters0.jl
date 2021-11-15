
γ1 = 0.3
γ0 = 0.4
η1 = 1.0#efficeincy
η0 = 1.0
n1 = 10.0
n0 = 1000.0
Γ1 = 0.8 #rate of measurment
Γ0 = 0.8


function parameters_matrix(γ,Γ,η,n,b)


    A = [  -0.5*γ      0   ;
            0       -0.5*γ ]#|> sparse


    D =  [ (γ*(n+0.5)+Γ)      0     ;
            0           (γ*(n+0.5)+Γ) ]# |> sparse

    C=[    sqrt(4*η*Γ)           0;
              0             sqrt(4*η*Γ)] #|>sparse

    Γ =[  0.      0. ;
          0.     0.] #|>sparse

    return (A,C,D,Γ,b);
end


x0=[0.,0.]
x1=[0., 0.]


σu0=n0+0.5+Γ0/γ0
σu1=n1+0.5+Γ1/γ1
    #variance in the stationary case.
σ11 = (sqrt(1+16.0*η1*Γ1*σu1/γ1)-1)*γ1/(8.0*η1*Γ1)
σ01 = (sqrt(1+16.0*η0*Γ0*σu0/γ0)-1)*γ0/(8.0*η0*Γ0)


σ0=[ σ01     0;
     0       σ01] #|>sparse

σ1=[ σ11      0;
          0          σ11 ]

u00 =(x0,σ0)
u01 =(x1,σ1)
b=[0.,0.]



u00 =(x0,σ0)
u01 =(x0,σ0)

dynamics_parameters0=parameters_matrix(γ0,Γ0,η0,n0,b)
dynamics_parameters1=parameters_matrix(γ1,Γ1,η1,n1,b)

#==
(A,C,D,Γ,b)=dynamics_parameters0

tmp =(σ0*(C')+Γ')
dσ = A*σ0+σ0*(A')+D-tmp*(tmp')
==#
