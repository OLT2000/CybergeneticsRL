module toggleswitchmodels

   using Catalyst, Trapz, DifferentialEquations, Statistics

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

   function hill_fn(var, theta, eta)
      return (1+(var/theta)^eta)^(-1)
   end

   function ts_model_convex(du, u, p, t)
      ctrl_var = p(u, t)

      aTc = 100*ctrl_var
      IPTG = 1 - ctrl_var

      du[1] = klm0+klm*hill_fn(u[4]*hill_fn(aTc,thetaAtc,etaAtc),thetaTet,etaTet)-glm*u[1]
      du[2] = ktm0+ktm*hill_fn(u[3]*hill_fn(IPTG,thetaIptg,etaIptg),thetaLac,etaLac)-gtm*u[2]
      du[3] = klp*u[1]-glp*u[3]
      du[4] = ktp*u[2]-gtp*u[4]

   end

   # Obtaining steady states and maxes
   const known_u0 = [2.55166047363230, 38.7108543679906, 102.155003051775, 1196.05604522200]

   ss_end = 1000
   sstspan = (0, ss_end)
   tempsteps = range(0, ss_end; step=1)
   tet_ctrl = 0

   ss_p = ODEProblem(ts_model_convex, known_u0, sstspan, (x,t)->tet_ctrl)
   max_tet_sln = solve(ss_p, Tsit5(), saveat = tempsteps)
   max_tet_ss = max_tet_sln[end]

   iptg_ctrl = 1
   ss_p = remake(ss_p, p=(x,t)->iptg_ctrl)
   max_lac_sln = solve(ss_p, Tsit5(), saveat=tempsteps)
   max_lac_ss = max_lac_sln[end]

   const lower_bounds = min.(max_lac_ss, max_tet_ss)
   const upper_bounds = max.(max_lac_ss, max_tet_ss)
   const state_range = upper_bounds .- lower_bounds

   lac_transcription = (u::Vector, i, maxa) -> klm0 + klm*hill_fn(u[4]*hill_fn(maxa*i, thetaAtc, etaAtc), thetaTet, etaTet)
   tet_transcription = (u::Vector, i, maxi) -> ktm0 + ktm*hill_fn(u[3]*hill_fn(maxi*(1-i), thetaIptg, etaIptg), thetaLac, etaLac)
   lac_translation = u::Vector -> klp*u[1]
   tet_translation = u::Vector -> ktp*u[2]
   ml_deg = u::Vector -> glm*u[1]
   mt_deg = u::Vector -> gtm*u[2]
   lac_deg = u::Vector -> glp*u[3]
   tet_deg = u::Vector -> gtp*u[4]

   export ts_model_convex, hill_fn, propensity_fn, get_statespace, lower_bounds, upper_bounds, state_range, get_rx_network, get_errors, max_tet_ss


   function propensity_fn(u::Vector, i; atcmax=100, iptgmax=1)
      prop_vec = [lac_transcription(u, i, atcmax), tet_transcription(u, i, iptgmax), lac_translation(u), tet_translation(u),
                  ml_deg(u), mt_deg(u), lac_deg(u), tet_deg(u)]
      return prop_vec
   end

   function get_errors(sol_data, tgt)
       error_data = tgt*ones(length(sol_data[4, :])) .- sol_data[4, :]
       MAE = mean(abs.(error_data))
       MSE = mean(error_data.^2)
       # ISE = trapz(error_data.^2, sol_data.t)
       # ITAE = trapz(sol_data.t.*(abs.(error_data)), sol_data.t)
       ISE = trapz(sol_data.t, error_data.^2)
       TAE = sol_data.t .* abs.(error_data)
       ITAE = trapz(sol_data.t, TAE)

       return ([MAE, MSE, ISE, ITAE], error_data)
   end

   function get_rx_network(atc_conc, iptg_conc)
      f1 = (tet) -> klm0 + klm*hill_fn(tet*hill_fn(atc_conc, thetaAtc, etaAtc), thetaTet, etaTet)
      f2 = (lac) -> ktm0 + ktm*hill_fn(lac*hill_fn(iptg_conc, thetaIptg, etaIptg), thetaLac, etaLac)

      rn = @reaction_network begin

         f1(TetR), 0 --> mRNA_l
         f2(LacI), 0 --> mRNA_t

         9.726e-1, mRNA_l --> mRNA_l + LacI
         1.170, mRNA_t --> mRNA_t + TetR

         1.386e-1, mRNA_l --> 0
         1.386e-1, mRNA_t --> 0
         1.65e-2, LacI --> 0
         1.65e-2, TetR --> 0

      end
      return rn
   end

   #Discretisations
   function get_statespace(n_states)
      global lower_bounds, state_range
       nbins = n_states - 2

       bin_widths = state_range./nbins

       icspace = copy(lower_bounds)
       for bin in 1:1:nbins
           icspace = hcat(icspace, bin.*bin_widths)
       end
       # Append the minima
       icspace = hcat(icspace, icspace[:, end].+lower_bounds)
       return icspace
   end
end  # module
