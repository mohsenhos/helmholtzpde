
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
import ModelingToolkit: Interval, infimum, supremum

@parameters x y t
@variables u(..)
Dx=Differential(x)^1
Dy=Differential(y)^1
Dxx = Differential(x)^2
Dyy = Differential(y)^2

# 2D PDE
eq  = Dxx(u(x,y)) + Dyy(u(x,y))+ u(x,y) ~ 0
#bb(t)=exp(complex(0,t))

# Boundary conditions

bcs = [Dxx(u(-1,y)) ~ -u(-1,y), Dxx(u(1,y)) ~ -u(1,y),
        Dyy(u(x,-1)) ~ -u(x,-1)+2,Dy(u(range(0, step=0.05, stop=1),0)) ~ 0,Dyy(u(x,1))~ -u(x,1)]
    #
# Space and time domains
domains = [x ∈ Interval(-1.0,1.0),
           y ∈ Interval(-1.0,1.0)]

# Neural network
dim = 2 # number of dimensions
chain = FastChain(FastDense(dim,8,Flux.σ),FastDense(8,8,Flux.σ),FastDense(8,1))
# Initial parameters of Neural network
initθ = Float64.(DiffEqFlux.initial_params(chain))

# Discretization
dx = 0.05
discretization = PhysicsInformedNN(chain,GridTraining(dx),init_params =initθ)

@named pde_system = PDESystem(eq,bcs,domains,[x,y],[u(x, y)])
prob = discretize(pde_system,discretization)

#Optimizer
opt = Optim.BFGS()

#Callback function
cb = function (p,l)
    println("Current loss is: $l")
    return false
end

res = GalacticOptim.solve(prob, opt, cb = cb, maxiters=1500)
phi = discretization.phi

using Plots

xs,ys = [infimum(d.domain):dx/10:supremum(d.domain) for d in domains]
analytic_sol_func(x,y) = y^2*sin(pi*x)

u_predict = reshape([first(phi([x,y],res.minimizer)) for x in xs for y in ys],(length(xs),length(ys)))
u_real = reshape([analytic_sol_func(x,y) for x in xs for y in ys], (length(xs),length(ys)))
diff_u = abs.(u_predict .- u_real)

p1 = plot(xs, ys, u_real, linetype=:surface,title = "analytic");
p2 = plot(xs, ys, u_predict, linetype=:surface,title = "predict");
p3 = plot(xs, ys, diff_u,linetype=:contourf,title = "error");
plot(p1,p2,p3)
