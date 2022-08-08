include("src\\FixedATCModels.jl")

using .fixedmodels
using DifferentialEquations, Plots, OhMyREPL
using Optimization, OptimizationOptimisers, Statistics, Trapz
using OptimizationBBO
using LinearAlgebra, JLD2

mutable struct controller
    kp
    ki
    kd
    past_e
    int_term
end #struct

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

function SDESim(x0,u0,t1, t2, dt,noise)
    stoich_matrix = [-1 0 0 0 0 0 1 0;0 -1 0 0 0 0 0 1;0 0 -1 0 1 0 0 0;0 0 0 -1 0 1 0 0]
    iptg = u0[1]
    fixed_atc=u0[2]
    simpts = t1:dt:t2
    xall = zeros(size(simpts)[1], 4)
    xall[1, :] = x0
    xlast = x0

    for j in t1:dt:(t2-dt)
        noise_idx = convert(Int, (j/dt) + 1)

        propensities = LugagnePropensities(xlast, iptg, fixed_atc)
        du = stoich_matrix * propensities

        da = stoich_matrix * Diagonal(sqrt.(propensities))
        dw = (noise[noise_idx+1, :]-noise[noise_idx, :])

        xlast = xlast + du.*dt + da*dw
        xlast = max(xlast, 0)
        xall[(noise_idx - t1/dt)+1, :] = xlast
    end

    return xall[2:end, :], simpts[2:end]
end

function pid_simulation(x0, ref, ctrlp, sample_t, totaltime, odeflag; savepts = 1.0)
    last_state = copy(x0)
    last_tet = last_state[4]
    atc = 25
    n_samples = totaltime/sample_t
    # ode = ODEProblem(fixedmodels.fixed_atc_model, last_state, (0.0, sample_t), (u, t)->[25, 1])
    ode = ODEProblem(fixedmodels.fixed_atc_model, last_state, (0.0, sample_t), (u, t)->[25, 1])

    ctrl = controller(ctrlp[1], ctrlp[2], ctrlp[3], 0.0, 0.0)

    all_data = copy(last_state)
    iptg_data = Vector{Float64}()

    sim = 1
    while sim <= n_samples
        e = ref - last_tet

        p = ctrl.kp * e
        if sim*sample_t >= 120
            i = ctrl.int_term + ctrl.ki*(e*sample_t)
            ctrl.int_term = i
        else
            i = 0
        end
        if sim > 1
            d = ctrl.kd*(e-ctrl.past_e)/sample_t
        else
            d = 0.0
        end

        u = sum([p, i, d])
        ctrl.past_e = e

        if u < 0.0
            u = 0.0
        else
            u = min(u, 1.0)
        end

        push!(iptg_data, u)

        if odeflag
            tempode = remake(ode, u0 = last_state, p = (x, t) -> [25, u])
            tempsol = solve(tempode, Tsit5(), saveat = savepts)

            all_data = hcat(all_data, tempsol[:, 2:end])
            last_state = tempsol[end]
        # else
        #     t1 = 5*(sim-1)
        #     t2 = 5*sim
        #     tempsol, tempt = SDESim(last_state, [25, u], t1, t2, 0.5, noise)
        #     all_data = hcat(all_data, tempsol[:, 2:end])
        end
        last_tet = last_state[4]

        sim+=1
    end

    timedata = 0.0:1.0:totaltime
    iptg_data = repeat(iptg_data, inner=sample_t)
    return all_data, timedata, iptg_data
end

function bangbang(x0, ref, samplet, runtime; savepts = 1)
    last_x = copy(x0)
    last_tet = last_x[4]
    atc = 25
    n_samples = runtime/samplet
    # ode = ODEProblem(fixedmodels.fixed_atc_model, last_state, (0.0, sample_t), (u, t)->[25, 1])
    ode = ODEProblem(fixedmodels.fixed_atc_model, last_x, (0.0, samplet), (u, t)->[25, 1])
    all_data = copy(last_x)
    iptg_data = Vector{Float64}()

    sim = 1
    while sim <= n_samples
        if last_tet > ref
            iptg = 0
        else
            iptg = 1
        end

        tempode = remake(ode, u0 = last_x, p = (x, t) -> [25, iptg])
        tempsol = solve(tempode, Tsit5(), saveat = savepts)

        push!(iptg_data, iptg)
        all_data = hcat(all_data, tempsol[:, 2:end])
        last_x = tempsol[end]
        last_tet = last_x[4]

        sim+=1
    end
    timedata = 0.0:savepts:runtime
    iptg_data = repeat(iptg_data, inner=samplet)
    return all_data, timedata, iptg_data
end




function pid_loss(ctrlp, p)
    x0, ref, sample_t, totaltime, savepts = p
    last_state = copy(x0)
    last_tet = last_state[4]
    atc = 25
    n_samples = totaltime/sample_t
    ode = ODEProblem(fixedmodels.fixed_atc_model, last_state, (0.0, sample_t), (u, t)->[25, 1])

    ctrl = controller(ctrlp[1], ctrlp[2], ctrlp[3], 0.0, 0.0)

    all_data = copy(last_state)

    sim = 1
    while sim <= n_samples
        e = ref - last_tet

        p = ctrl.kp * e
        if sim*sample_t >= 120
            i = ctrl.int_term + ctrl.ki*(e*sample_t)
            ctrl.int_term = i
        else
            i = 0
        end
        if sim > 1
            d = ctrl.kd*(e-ctrl.past_e)/sample_t
        else
            d = 0.0
        end

        u = sum([p, i, d])
        ctrl.past_e = e

        if u < 0.0
            u = 0.0
        else
            u = min(u, 1.0)
        end

        tempode = remake(ode, u0 = last_state, p = (x, t) -> [25, u])
        tempsol = solve(tempode, Tsit5(), saveat = savepts)

        all_data = hcat(all_data, tempsol[:, 2:end])
        last_state = tempsol[end]
        last_tet = last_state[4]

        sim+=1
    end

    teterror = all_data[4, :] .- ref
    loss = sum(abs2, teterror)
    return loss

end

function pid_error_metrics(sol, time, tgt; verbose = true)
   error_data = sol[4, :] .- tgt
   MAE = mean(abs.(error_data))
   MSE = mean(error_data.^2)
   IAE = trapz(time, abs.(error_data))
   ISE = trapz(time, error_data.^2)
   TAE = time .* abs.(error_data)
   ITAE = trapz(time, TAE)
   if verbose
     @printf("MAE: %g || MSE: %g || IAE: %.5g || ISE: %.5g || ITAE: %.5g\n", MAE, MSE, IAE, ISE, ITAE)
  end
   return ([MAE, MSE, IAE, ISE, ITAE], error_data)
end

ng_data = []

for n in 1:1:10
    fname = "julia_bits\\noise_grids\\500noisegrid_"*string(n)
    ng = JLD2.load_object(fname)
    push!(ng_data, ng)
end


y0 = copy(fixedmodels.max_tet)
goal = y0[4]/2
sample = 5
total = 500

ng1 = mapreduce(permutedims, vcat, ng_data[1].u)

soln, timedata, inputs = pid_simulation(y0, goal, [0.0320596, 0.000807411, 0.0764627], sample, 500, true)

p1 = plot(timedata, soln[4, :], linewidth = 3, label = "TetR", ylim = (0, 1200), ylabel = "Expression [a.u.]", xlabel = "Time [min]")
p1 = plot!(timedata, soln[3, :], linewidth = 3, label = "LacI")
plot(1:501, fill(goal, 501), style = :dash, color = :black, label = "aTc = 10")
# a, b, c = bangbang(y0, goal, sample, total)
# plot(b, a[4, :])

iter = 1
cb = function(p, l)
    global iter
    if iter % 2000 == 1
        print("Iteration: ")
        print(iter)
        print(" || Current Loss: ")
        println(l)
    end
    iter += 1
    return false
end

initp = [1.0, 1.0, 1.0]
params = [y0, goal, sample, total, 1.0]

pidprob = OptimizationProblem(pid_loss, initp, params, lb = [0.0, 0.0, 0.0], ub = [1000.0, 1000.0, 1000.0])
res = solve(pidprob, BBO_adaptive_de_rand_1_bin_radiuslimited(), callback = cb, maxiters = 8000)

testy0 = copy(fixedmodels.max_lac)


pid_error_metrics(soln, timedata, goal)

pi2 = plot(timedata[1:end-1], inputs, seriestype=:steppre, ylim=(0, 1), ylabel="IPTG input [mM]", xlabel="Time [min]", legend=false, linewidth = 3)

pt2 = plot(timedata, soln[4, :], ylabel="TetR", xlabel="Time [min]", legend=false)

plot(pt1, pi1, pt2, pi2, layout=(2, 2))
savefig("julia_bits\\final_figs\\PIDcontrol.pdf")
# Best candidate found: [0.0320596, 0.000807411, 0.0764627]
# Best candidate found: [1.71561, 0.00204841, 7.64178]
# Fitness: 6592427.337508647

# Best candidate found: [7.99553, 0.000232655, 33.0488]

# Best candidate found: [20.812, 0.0297942, 85.3393]
#
# Fitness: 6615185.159114621
