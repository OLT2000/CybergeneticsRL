module NN
   include("Models.jl")
   include("FixedATCModels.jl")
   using .toggleswitchmodels, .fixedmodels
   using DiffEqFlux

   export get_convex_agent, get_iptg_agent, get_noisy_iagent

   function get_noisy_iagent(network::DiffEqFlux.FastChain, n_inputs::Int, imax::Int, dist_arr; fixed_atc = 25)
      test_ps = initial_params(network)
      try
         network(ones(n_inputs), test_ps)
      catch
         @warn "Ensure the number of inputs matches your controller"
      end
      function noisy_agent(du, u, p, t)
         agn = [rand(d) for d in dist_arr]
         y = u[1:n_inputs] + agn
         nn_val = network(y, p)
         ctrl_input = nn_val[1]
         atc = fixed_atc
         iptg = ctrl_input*imax

         du[5] = atc
         du[6] = iptg
         fixed_atc_model(du, u[1:4], (u, t)->[atc, ctrl_input], t)

      end
      return noisy_agent
   end


   function get_iptg_agent_v2(network::DiffEqFlux.FastChain, n_inputs::Int, input_idx::Tuple{Int64, Int64}, imax::Int; fixed_atc = 25)
      test_ps = initial_params(network)
      try
         network(ones(n_inputs), test_ps)
      catch
         @warn "Ensure the number of inputs matches your controller"
      end

      function iptg_agent2(du, u, p, t)
         nn_val = network(u[[idx for idx in input_idx]], p)
         ctrl_input = nn_val[1]
         atc = fixed_atc
         iptg = ctrl_input*imax

         fixed_atc_model_v2(du, u[1:5], (u, t)->[atc, ctrl_input], t)

      end
      return iptg_agent2
   end


   function get_iptg_agent(network::DiffEqFlux.FastChain, n_inputs::Int, input_idx::Tuple{Int64, Int64}, imax::Int; fixed_atc = 25)
      test_ps = initial_params(network)
      try
         network(ones(n_inputs), test_ps)
      catch
         @warn "Ensure the number of inputs matches your controller"
      end

      function iptg_agent(du, u, p, t)
         nn_val = network(u[[idx for idx in input_idx]], p)
         ctrl_input = nn_val[1]
         atc = fixed_atc
         iptg = ctrl_input*imax

         du[5] = atc
         du[6] = iptg
         fixed_atc_model(du, u[1:4], (u, t)->[atc, ctrl_input], t)

      end

      return iptg_agent
   end

   function get_convex_agent(network::DiffEqFlux.FastChain, n_inputs::Int, amax::Int, imax::Int)
      test_ps = initial_params(network)
      try
         network(ones(n_inputs), test_ps)
      catch
         @warn "Ensure the number of inputs matches your controller"
      end

      function ts_agent(du, u, p, t)
         nn_val = network(u[1:n_inputs], p)
         ctrl_input = nn_val[1]
         atc = amax*ctrl_input
         iptg = (1 - ctrl_input)*imax

         du[5] = atc
         du[6] = iptg
         ts_model_convex(du, u[1:4], (u, t)->ctrl_input, t)

      end

      return ts_agent
   end

end #module
