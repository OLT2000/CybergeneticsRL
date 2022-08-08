module fixedmodels
    using DifferentialEquations

    export fixed_atc_model, statespace, max_tet, known_ic, ubs, fixed_atc_model_v2

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
   max_iptg = 1
   kin_iptg = 2.75e-2
   kout_iptg = 1.11e-1
   kin_atc = 1.62e-1
   kout_atc = 2.00e-2

   # Obtaining steady states and maxes
   known_ic = [2.55166047363230, 38.7108543679906, 102.155003051775, 1196.05604522200]

   function hill_fn(var, theta, eta)
      return (1+(var/theta)^eta)^(-1)
   end

   function fixed_atc_model(du, u, p, t)
      fixed_atc = p(u, t)[1]
      ctrl_var = p(u, t)[2]

      du[1] = klm0+klm*hill_fn(u[4]*hill_fn(fixed_atc,thetaAtc,etaAtc),thetaTet,etaTet)-glm*u[1]
      du[2] = ktm0+ktm*hill_fn(u[3]*hill_fn(ctrl_var*max_iptg,thetaIptg,etaIptg),thetaLac,etaLac)-gtm*u[2]
      du[3] = klp*u[1]-glp*u[3]
      du[4] = ktp*u[2]-gtp*u[4]

   end

   function fixed_atc_model_v2(du, u, p, t)
      fixed_atc = p(u, t)[1]
      input_iptg = p(u, t)[2]*max_iptg

      du[1] = klm0+klm*hill_fn(u[4]*hill_fn(fixed_atc,thetaAtc,etaAtc),thetaTet,etaTet)-glm*u[1]
      du[2] = ktm0+ktm*hill_fn(u[3]*hill_fn(u[5],thetaIptg,etaIptg),thetaLac,etaLac)-gtm*u[2]
      du[3] = klp*u[1]-glp*u[3]
      du[4] = ktp*u[2]-gtp*u[4]
      if input_iptg > u[5]
         du[5] = kin_iptg*(input_iptg - u[5])
      else
         du[5] = kout_iptg*(input_iptg - u[5])
      end
   end

   _p2 = ODEProblem(fixed_atc_model_v2, vcat(known_ic, 0), (0, 1000), (x,t)->[25, 1])
   max_tet2 = solve(_p2, Tsit5(), saveat = 1)[end]

   max_lac2 = solve(remake(_p2, p=(x,t)->[25, 0]), Tsit5(), saveat = 1)[end]

   lbs2 = min.(max_lac2, max_tet2)
   ubs2 = max.(max_lac2, max_tet2)
   bounds2 = ubs2 .- lbs2

   _p = ODEProblem(fixed_atc_model, known_ic, (0, 1000), (x,t)->[25, 1])
   max_tet = solve(_p, Tsit5(), saveat = 1)[end]

   max_lac = solve(remake(_p, p=(x,t)->[25, 0]), Tsit5(), saveat = 1)[end]

   lbs = min.(max_lac, max_tet)
   ubs = max.(max_lac, max_tet)
   bounds = ubs .- lbs

   #Discretisations
   function statespace2(n_states)
       nbins = n_states - 2

       bin_widths = bounds2./nbins

       icspace = copy(lbs2)
       for bin in 1:1:nbins
           icspace = hcat(icspace, bin.*bin_widths)
       end
       # Append the minima
       icspace = hcat(icspace, icspace[:, end].+lbs2)
       return icspace
   end

   #Discretisations
   function statespace(n_states)
       nbins = n_states - 2

       bin_widths = bounds./nbins

       icspace = copy(lbs)
       for bin in 1:1:nbins
           icspace = hcat(icspace, bin.*bin_widths)
       end
       # Append the minima
       icspace = hcat(icspace, icspace[:, end].+lbs)
       return icspace
   end
end # module
