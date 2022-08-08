include("src\\FixedATCModels.jl")
using .fixedmodels
using DifferentialEquations, Plots, DiffEqFlux, Random, Optimization, OptimizationOptimisers
using Printf, LaTeXStrings, Trapz, Statistics, JLD2, CSV, Tables
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
# QSS params
gp = 1.65e-2

function ts_model(du, u, p, t)

   function hill_fn(var, theta, eta)
      return (1+(var/theta)^eta)^(-1)
   end

   aTc = 25
   IPTG = p(u, t)

   du[1] = klm0+klm*hill_fn(u[4]*hill_fn(aTc,thetaAtc,etaAtc),thetaTet,etaTet)-glm*u[1]
   du[2] = ktm0+ktm*hill_fn(u[3]*hill_fn(IPTG,thetaIptg,etaIptg),thetaLac,etaLac)-gtm*u[2]
   du[3] = klp*u[1]-glp*u[3]
   du[4] = ktp*u[2]-gtp*u[4]

end

initial_u = [2.55166047363230,
                       38.7108543679906,
                       102.155003051775,
                       1196.05604522200]

endtime = 500
sstspan = (0, endtime)
tempsteps = range(0, endtime; step=1)

ss_p = ODEProblem(ts_model, initial_u, sstspan, (x,t)->1)

ss_sln = solve(ss_p, Tsit5(), saveat = tempsteps)

u0 = copy(ss_sln[end])
push!(u0, 1) # Append initial IPTG value to our ic

tsref = u0[4]/2

tspan = (0.0, 400.0)
N = 400
tsteps = range(tspan[1], tspan[2], length = N)

dt = (tsteps[end] - tsteps[1]) / N


#=
Neural Network is our controller
It takes the current state as an input
(lac conc, tet conc)
Returns a single output (aTc concentration) being the control decision
=#
init_iptg = 1

hidden_nodes = 8
ts_controller = FastChain(FastDense(4, hidden_nodes, tanh), FastDense(hidden_nodes, 1, sigmoid))

function ts_agent(du, u, p, t)
   ctrl_input = ts_controller(u[1:4], p)[1]
   ts_model(du, u[1:4], (u, t)->ctrl_input, t)
   du[5] = ctrl_input
end

ts_prob = ODEProblem(ts_agent, u0, tspan, nothing)
savedata = []
itersave = []

function predict(p)
   tmp_prob = remake(ts_prob, p=p)
   pred = solve(tmp_prob, Tsit5(), saveat = tsteps)
   Array(pred)
end

function loss_fn(p)
   sol = predict(p)
   loss = sum(abs2, sol[4, :] .- tsref)
   # loss = sum(abs2, sol[4, :])
   # loss = sol[4, end] - ref
   return loss, sol
end

iter = 0
callback = function (prm, lss, pred)
     global iter
     iter += 1
     if iter % 50 == 1
        @printf("Iteration: %d || Loss: %g \n", iter, lss)
        # sol = predict(prm)
        # push!(savedata, sol[4, :])
        # push!(itersave, iter)
    end
   return false
end

# function simulate(p)
#    tmp_prob = remake(ts_prob, p=p)
#    pred = solve(tmp_prob, Tsit5(), saveat = tsteps)
#    return pred
# end
#
# function simulate_ic(p, x0)
#    tmp_prob = remake(ts_prob, u0 = x0, p=p)
#    pred = solve(tmp_prob, Tsit5(), saveat = tsteps)
#    return pred
# end
#
function simulate_samples(p, ts, span, x0)
   npts = (span[2] - span[1])/ts
   sample_ode = ODEProblem(ts_model, x0, (0, ts), nothing)
   states = reshape(x0, 1, :)
   tpts = [span[1]]
   inputs = []
   xlast = x0
   i=0
   while i < npts
      iptginput = ts_controller(xlast[1:4], p)[1]
      push!(inputs, iptginput)
      tframe = (0, ts).+(i*ts)
      ode = remake(sample_ode, u0 = xlast, tspan = tframe, p = (u, t)->iptginput)
      odesol = solve(ode, Tsit5(), saveat = 1)
      outputx = mapreduce(permutedims, vcat, odesol.u[2:end, :])
      outputt = odesol.t[2:end]
      states = vcat(states, outputx)
      tpts = vcat(tpts, outputt)
      xlast = outputx[end, :]
      i+=1
   end
   return states, tpts, inputs
end


function sol_error_metrics(sol, time, tgt; verbose = true)
   error_data = tgt*ones(length(sol[:, 4])) .- sol[:, 4]
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
#
function ode_error_metrics(sol_data, tgt; verbose = true)
    error_data = tgt*ones(length(sol_data[4, :])) .- sol_data[4, :]
    MAE = mean(abs.(error_data))
    MSE = mean(error_data.^2)
    IAE = trapz(sol_data.t, abs.(error_data))
    ISE = trapz(sol_data.t, error_data.^2)
    TAE = sol_data.t .* abs.(error_data)
    ITAE = trapz(sol_data.t, TAE)
    if verbose
      @printf("MAE: %g || MSE: %g || IAE: %.5g || ISE: %.5g || ITAE: %.5g\n", MAE, MSE, IAE, ISE, ITAE)
   end
    return ([MAE, MSE, IAE, ISE, ITAE], error_data)
end


function gen_ic_sample_plots(params, rand_ics, timespan)
   n_ic = size(rand_ics, 2)
   tet_data = []
   lac_data = []
   j = 1
   while j <= n_ic
      test_state = rand_ics[:, j]
      newsoln, tdata, ~ = simulate_samples(params, 5, timespan, test_state)
      push!(tet_data, newsoln[:, 4])
      push!(lac_data, newsoln[:, 3])
      j += 1
   end
   return tet_data, lac_data
end


function gen_ic_sample_error(params, rand_ics, timespan, ref)
   n_ic = size(rand_ics, 2)
   error_vals = zeros(n_ic, 5)
   j = 1
   while j <= n_ic
      test_state = rand_ics[:, j]
      newsoln, tdata, ~ = simulate_samples(params, 5, timespan, test_state)
      er = sol_error_metrics(newsoln,tdata, ref, verbose=false)
      error_vals[j, :] = er[1]
      j += 1
   end
   avrg_error = sum(error_vals, dims = 1)./n_ic
   return avrg_error[3:5]
end

function get_states(ub, lb, n)
   @assert n >= 2 "check number of states is greater than 1"
   jump = (ub .- lb) ./ (n-1)

   states = mapreduce(permutedims, vcat, [lb + (n-1).*(jump) for n in range(1, n)])
   states[:, 2] = reverse(states[:, 2])
   states[:, 4] = reverse(states[:, 4])
   return states
end

function prepare_u0(n_, upper_limit)
    randu0 = rand(4, n_) .* upper_limit
    ics = vcat(randu0, rand(1, n_))
    return ics
end


atclb = copy(fixedmodels.lbs)
atcub = copy(fixedmodels.ubs)

lr = 0.01
opt = OptimizationOptimisers.Adam(lr)
adtype = Optimization.AutoZygote()
optfn = Optimization.OptimizationFunction((x, p) -> loss_fn(x), adtype)
opt_params = []
_p = []

for n_batches in 12:12
   global opt_params, atcub, atclb
   ts_pinit = initial_params(ts_controller)
   print("Total Batches: ")
   println(n_batches)
   minibatches = get_states(atcub, atclb, n_batches)
   batch_epochs = 800
   batch_no = 1
   while batch_no <= n_batches
      iter = 0
      print("Batch No: ")
      print(batch_no)
      ic_no = ((batch_no - 1) % n_batches) + 1

      print(" || Minibatch IC: ")
      ic = copy(minibatches[ic_no, :])
      println(ic)
      push!(ic, rand(1)[1])
      ts_prob = remake(ts_prob, u0 = ic, p = ts_pinit)
      optprob = Optimization.OptimizationProblem(optfn, ts_pinit)
      result = Optimization.solve(optprob, opt, callback = callback, maxiters = batch_epochs)
      ts_pinit = copy(result.u)
      batch_no += 1
   end
   push!(_p, ts_pinit)
end

p12_3 = _p[1]

p7 = _p[1]

p12_2 = _p[1]
p12 = _p[1]



p2 = _p[1]

p7 = _p[1]
p7 = params400[6]
x, _ = gen_ic_sample_plots(p12_3, r_ics, (0, 500))
gen_ic_sample_error(p7, r_ics, (0, 500), tsref)
plot(1:501, x)

JLD2.save_object("julia_bits\\final_nns\\finals_ps_450epochs", _p)
tp = deepcopy(ts_pinit)
save_params = deepcopy(opt_params)

n_traj = 30
r_ics = prepare_u0(n_traj, atcub)

p2 = params200[1]

params100 = JLD2.load_object("julia_bits\\final_nns\\finals_ps_100epochs")
params200 = JLD2.load_object("julia_bits\\final_nns\\finals_ps_200epochs")
params300 = JLD2.load_object("julia_bits\\final_nns\\finals_ps_300epochs")
params400 = JLD2.load_object("julia_bits\\final_nns\\finals_ps_400epochs")
params500 = JLD2.load_object("julia_bits\\final_nns\\finals_ps_500epochs")
params250 = JLD2.load_object("julia_bits\\final_nns\\finals_ps_250epochs")
params350 = JLD2.load_object("julia_bits\\final_nns\\finals_ps_350epochs")
params450 = JLD2.load_object("julia_bits\\final_nns\\finals_ps_450epochs")

function get_traj_error(nn_p, rics, span, ref)
   er_data = []
   for p in nn_p
      ers = gen_ic_sample_error(p, rics, span, ref)
      push!(er_data, ers)
   end
   return mapreduce(permutedims, vcat, er_data)
end

er100 = get_traj_error(params100, r_ics, tspan, tsref)
er200 = get_traj_error(params200, r_ics, tspan, tsref)
er300 = get_traj_error(params300, r_ics, tspan, tsref)
er400 = get_traj_error(params400, r_ics, tspan, tsref)
er500 = get_traj_error(params500, r_ics, tspan, tsref)
er250 = get_traj_error(params250, r_ics, tspan, tsref)
er350 = get_traj_error(params350, r_ics, tspan, tsref)
er450 = get_traj_error(params450, r_ics, tspan, tsref)

itae = [er200[:, 3], er250[:, 3], er300[:, 3], er350[:, 3], er400[:, 3], er450[:, 3], er500[:, 3]]

itae_vals = mapreduce(permutedims, vcat, itae)

itae_range = [maximum(v) - minimum(v) for v in itae].*10^(-6)/2

function my_std(samples)
    samples_mean = mean(samples)
    samples_size = length(samples)
    samples = map(x -> (x - samples_mean)^2, samples)
    samples_sum = sum(samples)
    samples_std = sqrt(samples_sum / (samples_size - 1))
    return samples_std
end
mapslices(my_std, itae_vals, dims = 1)
itae_std = vars.^(1/2)
vars = var(permutedims(itae_vals), dims = 2)
itae_std = std(itae_vals, dims = 1)

itae_vals = itae_vals.*(10^(-6))
mean_itae = vec(mean(itae_vals, dims = 1))
itae_vec = [itae_vals[:, i] for i in 1:size(itae_vals, 2)]

qvals = [quantile(x, [0.1, 0.9]) for x in itae_vec]
itaevar = [v[1] for v in diff.(qvals)]

plot(2:12, permutedims(itae_vals), label = ["200" "250" "300" "350" "400" "450" "500"], legend = :bottomright, xlim = (2, 12), xticks = 1:1:12, ylim = (0,3),
         xlabel = "Number of Minibatches", ylabel = L"ITAE $[\mathbf{\times{10^{-6}}}]$", linewidth = 3)

plot(2:12, mean_itae, ribbon = itae_std, fillalpha = 0.3, legend = false, xlim = (2, 12), xticks = 1:1:12, ylim = (0, 3),
         xlabel = "Number of Minibatches", ylabel = L"ITAE $[\mathbf{\times{10^{-6}}}]$", linewidth = 5)

savefig("julia_bits\\final_figs\\minibatch_ribbonplot.pdf")

tet, lac = gen_ic_sample_plots(p7, r_ics, (0, 500))

plot(1:501, tet, label = false, ylim = (0, 1200), xlabel = "Time [min]", ylabel = "TetR Concentration [a.u]", linewidth = 2)
plot!(range(0, 500, length = 501), fill(tsref, 501), label = "Ref", style=:dash, color=:black)
savefig("julia_bits\\final_figs\\400epochs_7_mb.pdf")

opt_p = params500[end]

gen_ic_sample_error(single_ps, r_ics, tspan, tsref)
# 2.77e+4…
# 4.31e+6…
# 3.31e+6…


optimal_ps = params200[1]
