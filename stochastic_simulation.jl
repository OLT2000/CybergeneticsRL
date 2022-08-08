include("src\\FixedATCModels.jl")

using .fixedmodels
using DifferentialEquations, LinearAlgebra, Plots, DiffEqFlux, JLD2, Statistics, Random
using DifferentialEquations.EnsembleAnalysis, Trapz
# using PlotlyJS

mutable struct pidcontroller
    kp
    ki
    kd
    past_e
    int_term
end #struct

# Lugagne Parameters
klm0=3.20e-2
klm=8.30
thetaAtc=11.65
etaAtc=2.00
thetaTet=30.00
etaTet=2.00
glm=1.386e-1
ktm0=1.19e-1
ktm=2.06
thetaIptg=9.06e-2
etaIptg=2.00
thetaLac=31.94
etaLac=2.00
gtm=1.386e-1
klp=9.726e-1
glp=1.65e-2
ktp=1.170
gtp=1.65e-2
kIPTGin=2.75e-2
kAtcin=1.62e-1
kIPTGout=1.11e-1
kAtcout=2.00e-2

function hill_fn(var, theta, eta)
   return (1+(var/theta)^eta)^(-1)
end

function LugagnePropensities(x,IPTG,Atc)
    #Initializing the propensity matrix
    A=zeros(8)
    #Evaluatng the propensity functions (From the article)
    A[1]=glm*x[1]
    A[2]=gtm*x[2]
    A[3]=glp*x[3]
    A[4]=gtp*x[4]
    A[5]=klp*x[1]
    A[6]=ktp*x[2]
    A[7]=klm0+klm*hill_fn(x[4]*hill_fn(Atc,thetaAtc,etaAtc),thetaTet,etaTet)
    A[8]=ktm0+ktm*hill_fn(x[3]*hill_fn(IPTG,thetaIptg,etaIptg),thetaLac,etaLac)

    return A
end

function LugagnePropAndD(t,X,Atc,IPTG,extAtc,extIPTG)
    At=LugagnePropensities(X,IPTG,Atc);

    if extIPTG>IPTG
        dIPTG=kIPTGin*(extIPTG-IPTG); #IPTG evolution (diffusive term as in the paper)
    else
        dIPTG=kIPTGout*(extIPTG-IPTG); #IPTG evolution (diffusive term as in the paper)
    end

    if extAtc>Atc

        dAtc=kAtcin*(extAtc-Atc);  #Atc Evolution
    else
        dAtc=kAtcout*(extAtc-Atc);  #Atc Evolution
    end
    return At,dIPTG,dAtc
end

stoich_matrix = [-1 0 0 0 0 0 1 0;
                 0 -1 0 0 0 0 0 1;
                 0 0 -1 0 1 0 0 0;
                 0 0 0 -1 0 1 0 0]

# ng_storage = []
ng_store = []
Random.seed!(2)
Random.seed!(1)

function create_grid(dt, span)
    global ng_store
    _ts = Array(span[1]:dt:(span[2]+dt))
    W = sqrt(dt)*randn(length(_ts), 8)
    Wsum = cumsum(vcat(zeros(1, 8), W[1:end-1, :]), dims = 1)
    W_vec = [vec(Wsum[el, :]) for el in 1:1:size(Wsum, 1)]
    push!(ng_store, W_vec)
    grid = NoiseGrid(_ts, W_vec)
    return grid
end

# ng_data = []
#
# for n in 1:1:10
#     fname = "julia_bits\\noise_grids\\500noisegrid_"*string(n)
#     ng = JLD2.load_object(fname)
#     push!(ng_data, ng)
# end

# y0 = copy(fixedmodels.max_lac)
y0 = copy(fixedmodels.max_tet)
ref = y0[4]/2
sde_controller = FastChain(FastDense(4, 4, tanh), FastDense(4, 1, sigmoid))
# p200 = JLD2.load_object("julia_bits\\final_nns\\finals_ps_200epochs")
# p2 = p200[1]
# trained_ps = JLD2.load_object("julia_bits\\parameters\\controller22")

fixed_atc = 25
ctrlp = [0.0320596, 0.000, 0.0764627]
pidctrl = pidcontroller(ctrlp[1], ctrlp[2], ctrlp[3], 0.0, 0.0)

function pid_control(u, t)
    global ref, pidctrl
    e = ref - u[4]
    p = pidctrl.kp * e
    if t >= 1000
        i = pidctrl.int_term + pidctrl.ki*(e*5)
        pidctrl.int_term = i
    else
        i = 0
    end
    if t > 1
        d = pidctrl.kd*(e-pidctrl.past_e)/5
    else
        d = 0.0
    end
    iptg = sum([p, i, d])
    pidctrl.past_e = e

    if iptg < 0.0
        iptg = 0.0
    else
        iptg = min(iptg, 1.0)
    end
    return [25, iptg]
end

get_control(u, t) = [fixed_atc, sde_controller(u[1:4], p12_2)[1]]
# get_control(u, t) = pid_control(u, t)
#
# init_iptg = get_control(y0, 0.0)
iptg0 = [25, 0]
iptg1 = deepcopy(iptg0)

n_traj = 10

# time range for the solver
dt = 0.5
tspan = (0.0, 2000.0)
interval = 1
simspace = range(tspan[1], tspan[2], step = 1)


# noisegrid = create_grid(dt, tspan)

function drift_fn(u, p, t)
    global iptg1
    fixed_input = p(u, t)[1]
    if t % 5.0 != 0
        controller_input = iptg1
    else
        controller_input = p(u, t)[2]
        iptg1 = controller_input
    end
    propensities = LugagnePropensities(u[1:4], controller_input, fixed_input)
    du = stoich_matrix * propensities
    return du
end

function g(u, p, t)
    try
        global iptg1
        _fixed_input = p(u, t)[1]
        if t % 0 != 5.0
            _controller_input = iptg1
        else
            _controller_input = p(u, t)[2]
            iptg1 = controller_input
        end
        propensities = LugagnePropensities(u[1:4], _controller_input, _fixed_input)
        diag = Diagonal(sqrt.(propensities))
        dw = stoich_matrix * diag
        return dw
    catch e
        return zeros(4, 8)
    end
end

ng_store

# ng1 = create_grid(dt, tspan)
sde = SDEProblem(drift_fn, g, y0, tspan, (x,t)->get_control(x, t), noise = nothing, noise_rate_prototype=zeros(4, 8))

function prob_func(prob, i, repeat)
    # global ng_data
    global dt, tspan
    iptg1 = iptg0
    # new_grid = ng_data[i]
    new_grid = create_grid(dt, tspan)
    prob = remake(sde, noise = new_grid)
end

j = 1
input_stops = tspan[1]:5.0:tspan[2]
iptg_array = zeros(length(input_stops) - 1, n_traj)
condition(u, t, integrator) = t % 5 == 0
function affect!(integrator)
    global iptg_array, j
    iptg = get_control(integrator.u, integrator.t)[2]
    # iptg = pid_control(integrator.u, integrator.t)[2]
    iptg_array[j] = iptg
    j+=1
end
cb = DiscreteCallback(condition, affect!, save_positions = (false, false))

ensemble = EnsembleProblem(sde, prob_func = prob_func)
println("Solving Ensemble")
sim = solve(ensemble, EM(), trajectories = n_traj, isoutofdomain = (m,p,t) -> any(x->x<0, m),
    saveat = simspace, dt = dt, callback = cb, tstops=input_stops)

ptet = Plots.plot(sim, vars = (0, 4), linelpha=0.6, xlabel = "Time [min]", ylabel = "TetR Concentration [a.u]", ylim = (0, 1200), showlegend=false, label = false)
Plots.plot!(simspace, fill(ref, length(simspace)), label = "Ref", style = :dash, color = :black, legend = true)





Plots.savefig("julia_bits\\figures\\fixed_stoch\\multilineplotnn_v3_2000.pdf")

summ = EnsembleSummary(sim)

Plots.plot(summ; idxs=4, xlabel = "Time [min]", ylabel = "TetR Concentration [a.u.]", label = false, showlegend=false, ylim = (0, 1200))

Plots.savefig("julia_bits\\final_figs\\meanplot4d_1.pdf")


mean_input = vec(mean(iptg_array, dims = 2))
var_input = var(iptg_array, dims = 2)
stepmean = repeat(mean_input, inner = 5)

Plots.plot(simspace[1: end - 1], stepmean, seriestype = :steppre, ylabel = "IPTG Input", xlabel = "Time [min]", ylim = (0, 1), legend = false)
Plots.savefig("julia_bits\\final_figs\\meaninput_4d_1.pdf")

tetrmat = sim[4, :, :]


target = y0[4]/2
errormat = (tetrmat .- target)

maemat = sum(abs, errormat, dims = 1)
meanMAE = mean(maemat)
msemat = sum(abs2, errormat, dims = 1)
meanMSE = mean(msemat)

newtrapz = arr -> trapz(simspace, arr)
isemat = mapslices(newtrapz, errormat.^2, dims = 1)
iaemat = mapslices(newtrapz, abs.(errormat), dims = 1)
meanIAE = mean(iaemat)

meanISE = mean(isemat)
timeweight = x -> x .* simspace
taefn = mapslices(timeweight, errormat, dims = 1)
itaemat = mapslices(newtrapz, taefn, dims = 1)
meanITAE = mean(itaemat)

function error_metrics(error_data, time)
    MAE = mean(abs.(error_data))
    MSE = mean(error_data.^2)
    ISE = trapz(time, error_data.^2)
    IAE = trapz(time, abs.(error_data))
    TAE = time .* abs.(error_data)
    ITAE = trapz(time, TAE)
    return [MAE, MSE, IAE, ISE, ITAE]
end


error_vals = zeros(n_traj, 5)

for i in 1:1:n_traj
    arr = error_metrics(errormat[:, i], simspace)
    error_vals[i, :] = arr
end

mean(error_vals, dims = 1)
