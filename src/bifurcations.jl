using Revise, ForwardDiff, Parameters, Setfield, Plots, LinearAlgebra
using BifurcationKit, DifferentialEquations
const BK = BifurcationKit
using Plots.PlotMeasures

# sup norm
norminf(x) = norm(x, Inf)
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
fixed_atc = 25
kin_iptg = 2.75e-2
kout_iptg = 1.11e-1

function hill_fn(var, theta, eta)
   return (1+(var/theta)^eta)^(-1)
end

function fixed_atc_bif_v2(du, u, p, t)
	@unpack fixed_atc, input_iptg = p

	du[1] = klm0+klm*hill_fn(u[4]*hill_fn(fixed_atc,thetaAtc,etaAtc),thetaTet,etaTet)-glm*u[1]
	du[2] = ktm0+ktm*hill_fn(u[3]*hill_fn(input_iptg,thetaIptg,etaIptg),thetaLac,etaLac)-gtm*u[2]
	du[3] = klp*u[1]-glp*u[3]
	du[4] = ktp*u[2]-gtp*u[4]

end

function fixed_lugagne(u, p)
        @unpack fixed_atc, input_iptg = p
                mRNA_L, mRNA_T, LacI, TetR = u
                [
                        klm0+klm*hill_fn(TetR*hill_fn(fixed_atc,thetaAtc,etaAtc),thetaTet,etaTet)-glm*mRNA_L
                        ktm0+ktm*hill_fn(LacI*hill_fn(input_iptg,thetaIptg,etaIptg),thetaLac,etaLac)-gtm*mRNA_T
                        klp*mRNA_L-glp*LacI
                        ktp*mRNA_T-gtp*TetR
                ]

end

function fixed_lugagne_v2(u, p)
        @unpack fixed_atc, input_iptg = p
                mRNA_L, mRNA_T, LacI, TetR, iptg = u
				if input_iptg > iptg
					diptg = kin_iptg*(input_iptg - iptg)
				else
					diptg = kout_iptg*(input_iptg - iptg)
				end
                [
                        klm0+klm*hill_fn(TetR*hill_fn(fixed_atc,thetaAtc,etaAtc),thetaTet,etaTet)-glm*mRNA_L
                        ktm0+ktm*hill_fn(LacI*hill_fn(input_iptg,thetaIptg,etaIptg),thetaLac,etaLac)-gtm*mRNA_T
                        klp*mRNA_L-glp*LacI
                        ktp*mRNA_T-gtp*TetR
						diptg
                ]

end


# Obtaining steady states and maxes
known_ic = [2.55166047363230, 38.7108543679906, 102.155003051775, 1196.05604522200]
ic_v2 = [2.55166047363230, 38.7108543679906, 102.155003051775, 1196.05604522200, 0.0]
atcvar = 10
atcarr = range(10,45,6)

tetplots = []

for atc in [10, 17, 24, 31, 38, 45]
	parlug = (fixed_atc = atc, input_iptg = 0.0)
	ss_prob = SteadyStateProblem(fixed_atc_bif_v2, known_ic, p = parlug)
	ss_sol = solve(ss_prob, DynamicSS(Rodas5()))

	bifprob = BifurcationProblem(fixed_lugagne, ss_sol.u, parlug, (@lens _.input_iptg);
	                        recordFromSolution = (x, p) -> (TetR = x[4], LacI = x[3]))

	bif_opts = ContinuationPar(pMin = 0.0, pMax = 1.0, ds = 0.01, dsmax = 0.2, dsmin = 0.01, nev = 4, maxSteps = 10000,
					detectBifurcation=3)

	# compute the branch of solutions
	br = @time continuation(bifprob, PALC(), bif_opts;
		normC = norminf,
		bothside = true)

	p1 = plot(br, plotfold=true, markersize=2, title = "aTc = "*string(atc), label = false, plotstability=true, ylim=(0, 1200), #yguidefontsize=14, xguidefontsize=14,
		ylabel="TetR [a.u.]", xlabel="IPTG Input [mM]", linewidthunstable = 8, linewidthstable=1, linecolor=:blue)
	push!(tetplots, p1)
end

plot(tetplots[1], tetplots[2], tetplots[3], tetplots[4], tetplots[5], tetplots[6], size = (1400, 800), leftmargin=20px, bottommargin=20px, rightmargin=20px, legend=false)

p2 = plot(br, vars=(:param, :LacI), plotfold=true, markersize=2, legend=false, plotstability=true, ylim=(-200, 3400), yguidefontsize=14, xguidefontsize=14,
		ylabel="LacI [a.u.]", xlabel="IPTG Input [mM]", linewidthunstable = 5, linewidthstable=1, linecolor=:green)

p3 = plot(p1, p2, layout = (1, 2), size = (600, 250), yguidefontsize=8, xguidefontsize=8, rightmargin=15px, bottommargin=15px)

savefig("julia_bits\\figures\\finalbif.pdf")
