module SSA
    using Random, DifferentialEquations, LinearAlgebra
    include("Models.jl")
    using .toggleswitchmodels

    mutable struct output
        t::Vector{Float64}
        s::Matrix{Float64}
        i::Vector{Float64}
    end

    export firstrxmethod, fullscale_rx_sim, forcing_fn!, noise_fn!

    const stoich_matrix = [1 0 0 0 -1 0 0 0;
                           0 1 0 0 0 -1 0 0;
                           0 0 1 0 0 0 -1 0;
                           0 0 0 1 0 0 0 -1]

    function forcing_fn(u, p, t)
      controller_input = p(u, t)
      propensities = toggleswitchmodels.propensity_fn(u[1:4], controller_input)
      du = stoich_matrix * propensities
      return du
    end

    function noise_fn(u, p, t)
        try
            controller_input = p(u, t)
            propensities = propensity_fn(u[1:4], controller_input)
            diag = Diagonal(sqrt.(propensities))
            du = stoich_matrix * diag
            return du
        catch
            du = fill(Inf, 4, 8)
            return du
        end
    end

    function reaction_simulation(prev_state, control_input; input_time=15)
        atc_input = 100*control_input
        iptg_input = 1 - control_input
        rx_net = get_rx_network(atc_input, iptg_input)
        rx_sde = SDEProblem(rx_net, prev_state, (0, input_time))
        rx_sol = solve(rx_sde, RKMilCommute(); saveat = 1, isoutofdomain=(u,p,t) -> any(x->x<0, u))
        return rx_sol
    end

    function fullscale_rx_sim(initial_u, control_function, n_samples; input_t = 15)
        temp_state = initial_u
        temp_inducer = control_function(initial_u[1:4])
        tdata = [1.0]
        sdata = reshape(initial_u, 1, :)
        idata = [temp_inducer]
        for i in 1:1:n_samples
            temp_sol = reaction_simulation(temp_state, temp_inducer; input_time = input_t)
            new_data = mapreduce(permutedims, vcat, temp_sol.u[2:end])
            sdata = vcat(sdata, new_data)
            updated_t = temp_sol.t .+ (i*input_t)
            tdata = vcat(tdata, updated_t[2:end])

            # reassign values
            idata = vcat(idata, temp_inducer .* ones(length(temp_sol[2:end])))

            temp_state = sdata[end, :]
            temp_inducer = control_function(temp_state)
        end

        full_output = output(tdata, sdata, idata)

        return full_output
    end

    function firstrxmethod(x0::Vector, ctrl_fn, propensities, stch_matrix::Matrix, limit::Int; maxrx = true, maxtime = false)
        @assert xor(maxrx, maxtime) "Define the limit as a time limit or reaction limit, not both"
        @assert (typeof(x0)==Vector{Int}) "Ensure concentrations are integers corresponding to N molecules"
        rx = 1
        time = [0.0]

        prev_input_t = 0.0
        input_t = 15.0

        ctrl = ctrl_fn(x0)

        concs = reshape(x0, 1, :)

        println("Starting simulation.")
        running = true
        while running
            local rj

            curr_state = concs[rx, :]

            # Obtain input on sample intervals
            if ((time[end] - prev_input_t) >= input_t)
                ctrl = ctrl_fn(curr_state)
                prev_input_t = time[end]
            end

            aj = propensities(curr_state, ctrl)

            if all(r==0 for r in aj)
                println("all rates are zero")
                running = false
            end

            Σaj = sum(aj)

            reaction_probs = [rt/Σaj for rt in aj]

            rj = rand(size(stch_matrix)[2])
            tauj = (-log.(rj))./aj

            tau_min = minimum(tauj)
            idx = argmin(tauj)

            updates = stch_matrix[:, idx]
            new_state = curr_state + updates

            push!(time, time[end]+tau_min)
            concs = vcat(concs, reshape(new_state, 1, :))

            if (maxrx & (rx == limit))
                println("Maximum reactions reached. Terminating.")
                running = false
            elseif (maxtime &  (time[end] >= limit))
                println("Maximum time exceeded. Termminating")
                running = false
            else
                rx += 1
            end
        end

        return time, concs
    end

end #module
