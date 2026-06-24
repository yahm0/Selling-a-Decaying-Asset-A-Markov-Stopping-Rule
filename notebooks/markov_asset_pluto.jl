### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #= bind fallback for running outside Pluto =#
    quote
        local iv = try
            Base.loaded_modules[Base.PkgId(
                Base.UUID("6e696c72-6542-2067-7265-42206c756150"),
                "AbstractPlutoDingetjes",
            )].Bonds.initial_value
        catch
            ;
            b -> missing;
        end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 980248fd-ee83-44f3-9784-592148b383a2
md"""
# The Markov-Modulated Asset: Optimal Stopping on a Markov Chain (Pluto)

**Optimal Stopping + Discrete-Time Finite Markov Chains**

An asset has $M$ quality states $S = \{0, \dots, M-1\}$ ($0$ = distressed, $M-1$ = pristine).
Quality evolves each period by a transition matrix $P$. Selling in state $s$ pays $R(s)$;
holding one period costs $c$; sale is forced at horizon $T$. The value function obeys the Bellman recursion

$$V_t(s) = \max\Big(\; R(s)\;,\; -c + \sum_{j \in S} P_{sj}\, V_{t+1}(j) \;\Big), \qquad V_T(s) = R(s).$$

The first branch is **SELL** (stop and collect), the second is **HOLD** (a row of $P$ prices tomorrow).
Because the state space and horizon are finite, backward induction gives the *exact* solution.

Formally, this is a finite-horizon **Markov-modulated optimal stopping** model: a finite Markov chain
modulating an optimal stopping decision. This is a reactive Pluto port of the same model, matrix,
parameters, and results as the IJulia notebook.
"""

# ╔═╡ 8de70c43-145b-4bde-b0da-3fc3b9ffef71
md"""
## Before the math: three layers, kept separate

1. **The model** answers *"what world are we in?"* Five quality levels that hop around by Markov chain
   rules, a price tied to quality, a holding fee each period, and a hard deadline. The goal: pick a
   stopping time $\tau$ to maximize expected profit, $\mathbb{E}[R(S_\tau) - c\,\tau]$.
2. **The rule (the Bellman equation)** answers *"what does optimal mean here?"* At every state and time,
   the best you can do is the better of two numbers: cash now, or the average of tomorrow's best, minus
   the fee.
3. **The recipe (backward induction)** answers *"how do we compute it?"* On the last day the choice is
   forced, so the answer there is known. Walk backward one period at a time and the whole decision table
   fills itself in.
"""

# ╔═╡ 5e30e618-d1ca-4884-a5da-1613e17aa785
begin
    # Use the repo-root environment (Project.toml/Manifest.toml one level up) as the
    # single source of truth. Calling Pkg.activate also tells Pluto not to use its own
    # built-in package manager, so this notebook shares versions with the IJulia one.
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    using LinearAlgebra   # eigen, eigvals, dot, norm, I
    using Printf          # @printf formatted output
    using Plots           # figures (GR backend by default)
    using PlutoUI         # Slider, CheckBox, TableOfContents for interactivity
end

# ╔═╡ 7d876aa4-d163-410d-b870-ada9c1b0e4d8
TableOfContents()

# ╔═╡ 4cc7dfa0-f937-4b61-ac2e-99ad733c532e
# Figures are written to the repo-root figures/ folder (one level up from notebooks/).
FIGDIR = joinpath(@__DIR__, "..", "figures")

# ╔═╡ 1b2293a9-15b1-4547-90ba-bcfb58d32c8b
md"""
## 1. The Markov environment

`MarkovAsset` validates $P$ (rows sum to 1, non-negative), computes the stationary distribution $\pi$
from the left eigenvector for eigenvalue 1 ($\pi P = \pi$), and measures how fast $P^n$ approaches $\pi$.
The asymptotic rate is the **second-largest eigenvalue modulus** $|\lambda_2|$: the chain forgets its
starting state geometrically, like $|\lambda_2|^n$.
"""

# ╔═╡ 3bf39c65-4da1-4f17-a5ff-f048939de668
begin
    struct MarkovAsset
        P::Matrix{Float64}
        M::Int
        function MarkovAsset(P::AbstractMatrix; atol = 1e-10)
            P = Float64.(P)
            @assert size(P, 1) == size(P, 2) "P must be square"
            @assert all(P .>= -atol) "P must have non-negative entries"
            @assert all(isapprox.(sum(P; dims = 2), 1.0; atol = atol)) "rows must sum to 1"
            new(P, size(P, 1))
        end
    end

    # Local name `pi` shadows Base's numeric constant to match the math; never needed here.
    function stationary_distribution(a::MarkovAsset)
        F = eigen(permutedims(a.P))
        idx = argmin(abs.(F.values .- 1.0))
        pi = real.(F.vectors[:, idx])
        pi ./= sum(pi)
        @assert all(pi .> 0) "chain must be irreducible (pi > 0)"
        return pi
    end

    slem(a::MarkovAsset) = sort(abs.(eigvals(a.P)); rev = true)[2]

    function convergence_profile(a::MarkovAsset; n_max = 50)
        pivec = stationary_distribution(a)
        dists = zeros(n_max)
        Pn = Matrix{Float64}(I, a.M, a.M)
        for n in 1:n_max
            Pn = Pn * a.P
            dists[n] = maximum(norm(Pn[i, :] .- pivec) for i in 1:a.M)
        end
        return dists
    end
end

# ╔═╡ 59e6b88a-9d2d-4581-9069-f5632e6954dc
md"""
## 2. The optimal stopping engine

Backward induction: set $V_T = R$, then walk back $t = T-1, \dots, 0$, comparing the immediate reward
against the expected continuation value. The policy matrix records the decision (1 = SELL, 0 = HOLD) at
every $(t, s)$. Where an i.i.d. offer model produces one threshold per time step, here the expectation
$\sum_j P_{sj} V_{t+1}(j)$ depends on the current state, so the rule becomes a boundary in the $(t, s)$ plane.
"""

# ╔═╡ 36384f99-29ce-4051-97f3-e5265dba74ab
begin
    struct DynamicStoppingSolver
        asset::MarkovAsset
        R::Vector{Float64}
        c::Float64
        T::Int
        function DynamicStoppingSolver(asset::MarkovAsset, R, c, T)
            R = Float64.(R)
            @assert length(R) == asset.M "R must give one payoff per state"
            @assert c >= 0 && T >= 1
            new(asset, R, Float64(c), Int(T))
        end
    end

    function solve(s::DynamicStoppingSolver)
        M, P, R, c, T = s.asset.M, s.asset.P, s.R, s.c, s.T
        # Julia is 1-indexed: row t+1 holds the value at time t, for t = 0..T.
        V = zeros(T + 1, M)
        policy = ones(Int, T + 1, M)                 # forced sale at t = T
        V[T + 1, :] .= R
        for t in (T - 1):-1:0
            continuation = -c .+ P * V[t + 2, :]     # value at time t+1 lives in row t+2
            V[t + 1, :] .= max.(R, continuation)
            policy[t + 1, :] .= Int.(R .>= continuation)   # tie-break: sell
        end
        return V, policy
    end
end

# ╔═╡ 8eea75fd-9705-429f-a33f-cb0486a2f198
md"""
## 3. Experiment

The chain has **recovery pressure at the bottom** (distressed assets get fixed up) and **decay pressure at
the top** (pristine assets wear down). Payoffs are $R(s) = 20(s+1)$ in \$k, holding costs $c = 2$ per
period, deadline $T = 12$.
"""

# ╔═╡ 0154692c-c4e2-474d-8385-6b6e3fb7ea2a
md"""
### Controls

Holding cost c = $(@bind c_ui Slider(0:0.25:10; default=2.0, show_value=true))

Horizon T = $(@bind T_ui Slider(1:30; default=12, show_value=true))

Payoff step (R(s) = step times (s+1)) = $(@bind step_ui Slider(5:5:60; default=20, show_value=true))

Highlight start state = $(@bind s0_ui Slider(0:4; default=0, show_value=true))

Export current figures to figures/ = $(@bind do_export CheckBox(default=false))
"""

# ╔═╡ 097ffedd-c828-4731-be05-1f978100c9e7
begin
    P = [
        0.50 0.40 0.10 0.00 0.00;
        0.20 0.45 0.30 0.05 0.00;
        0.05 0.25 0.45 0.20 0.05;
        0.00 0.10 0.30 0.45 0.15;
        0.00 0.05 0.15 0.35 0.45
    ]
    asset = MarkovAsset(P)
    R = Float64[step_ui * (s + 1) for s in 0:(asset.M - 1)]
    c = float(c_ui)
    T = Int(T_ui)
    pivec = stationary_distribution(asset)
    println("stationary pi      : ", round.(pivec; digits = 4))
    @printf("|lambda_2| (SLEM)  : %.4f\n", slem(asset))
    @printf("long-run avg R     : %.2f\n", dot(pivec, R))
end

# ╔═╡ 807262fd-22c4-44a3-84e4-f5010e8ec043
begin
    solver = DynamicStoppingSolver(asset, R, c, T)
    V, policy = solve(solver)
    println("V_0 by state: ", round.(V[1, :]; digits = 2))
    policy
end

# ╔═╡ cb35f633-3a49-49a5-9b36-6b990f607fe8
md"""
**Highlighted start.** From state $(s0_ui) at t = 0 the optimal action is **$(policy[1, s0_ui + 1] == 1 ? "SELL" : "HOLD")**, with value V_0 = $(round(V[1, s0_ui + 1]; digits=2)) (\$k).
"""

# ╔═╡ ca246cb7-c3d6-4235-8ea9-b189302e4fc9
md"""
The value function at $t = 0$ is exact: no simulation is needed, because backward induction enumerates
every state and time. Averaging $V_0$ over a uniformly random starting state and comparing against selling
immediately quantifies the value of optimal timing.
"""

# ╔═╡ 2d685b8c-9916-43a2-884d-3628f33a8976
begin
    uniform = fill(1 / asset.M, asset.M)
    optimal = dot(uniform, V[1, :])
    naive = dot(uniform, R)              # sell at t = 0 regardless of state
    @printf("expected profit, optimal rule : %.2f\n", optimal)
    @printf("expected profit, sell at t=0  : %.2f\n", naive)
    @printf(
        "value of timing               : %+.2f  (%.1f%%)\n",
        optimal - naive,
        (optimal/naive - 1)*100
    )
end

# ╔═╡ 117c3012-8b58-4107-8d4e-51bbcd1906e6
md"""
## 4. Convergence of $P^n$ to $\pi$

How far is the worst-case row of $P^n$ from the stationary distribution? The dashed reference shows the
geometric envelope $|\lambda_2|^n$, and the match is exact: $|\lambda_2| = 0.680$ is the true forgetting
rate. Memory of the initial state is effectively gone within ~10 periods, which is why stopping decisions
only matter early.
"""

# ╔═╡ 79d9497e-9df5-4fb4-86ee-6b82e6a6acad
begin
    dists = convergence_profile(asset; n_max = 40)
    lam2 = slem(asset)
    nn = 1:40

    plt_conv = plot(
        nn,
        dists;
        yscale = :log10,
        marker = :circle,
        ms = 3,
        lw = 1.8,
        color = :steelblue,
        label = "max_i ||(P^n)_i. - pi||_2",
        xlabel = "n (matrix power)",
        ylabel = "L2 distance to pi (log scale)",
        title = "Convergence of P^n to the stationary distribution",
        legend = :topright,
    )
    plot!(
        plt_conv,
        nn,
        dists[1] .* lam2 .^ (nn .- 1);
        ls = :dash,
        lw = 1.5,
        color = :firebrick,
        label = "geometric rate |lambda_2|^n, |lambda_2| = $(round(lam2; digits=3))",
    )
    plt_conv
end

# ╔═╡ e34d34e6-7aef-4944-b6b1-93c99c4b60fe
md"""
## 5. The decision boundary

Time on the x-axis, state on the y-axis. Orange = SELL, blue = HOLD. The optimal rule is **monotone**:
SELL iff $s \ge s^*(t)$, with $s^* = 3$ for $t \le 7$, $s^* = 2$ for $8 \le t \le 11$, and $s^* = 0$ at the
forced-sale deadline. The boundary steps *down* as the deadline nears, because patience loses value when
few recovery chances remain.
"""

# ╔═╡ 8d000d41-440c-43dc-b8d0-ac4e4db54fa4
begin
    grid = permutedims(policy)              # M x (T+1): rows = state, cols = time
    plt_policy = heatmap(
        0:T,
        0:(asset.M - 1),
        grid;
        color = cgrad([:steelblue, :sandybrown]),
        clims = (0, 1),
        colorbar = false,
        xlabel = "time t",
        ylabel = "quality state s",
        title = "Optimal policy: SELL (orange) vs HOLD (blue)",
        xticks = 0:T,
        yticks = 0:(asset.M - 1),
    )
    for t in 0:T, s in 0:(asset.M - 1)
        sell = policy[t + 1, s + 1] == 1
        annotate!(plt_policy, t, s, text(sell ? "S" : "H", 9, sell ? :black : :white))
    end
    plt_policy
end

# ╔═╡ 3c253ac7-0cc5-4066-9a1c-65c9893424da
begin
    do_export && savefig(plt_conv, joinpath(FIGDIR, "convergence.png"))
    do_export && savefig(plt_policy, joinpath(FIGDIR, "policy_heatmap.png"))
    do_export ? md"Figures exported to `figures/`." :
    md"Tick **Export current figures** above to save the current plots into `figures/`."
end

# ╔═╡ 67b80280-47b7-45a2-848c-e8ac95e8f210
md"""
## 6. Interpretation and limitations

**Answer.** Following the threshold rule, an asset of uniformly random quality is worth $\approx 71.1$ (\$k),
an 18% gain over selling immediately. High states liquidate at once; low states hold and wait for recovery.

**Division of labor.** The Markov chain predicts the environment ($P$, $\pi$, $|\lambda_2|$) but cannot decide anything;
The optimal-stopping rule turns those predictions into actions but is blind to tomorrow once offers aren't i.i.d.

**Binding limitation.** $P$ is assumed known and time-homogeneous; a misspecified or shifting $P$ moves the
boundary and erodes the edge.

**Out of reach.** If buyers react strategically to our selling rule, no single-agent optimum exists. That is
a game-theory question, unanswerable with any amount of data or compute.
"""

# ╔═╡ d4928829-8194-4e0a-a2f1-d6ada59f284b
md"""
## Verification against the reference

Every line below should print `true`. Any `false` is flagged with the computed-versus-reference numbers.
"""

# ╔═╡ 41612f26-8f50-46bb-991f-e593993cf44d
begin
    ref_pi = [0.1445, 0.2849, 0.3051, 0.1869, 0.0787]
    ref_lam2 = 0.6795
    ref_piR = 55.41
    ref_V0 = [53.35, 57.25, 64.68, 80.0, 100.0]
    ref_opt = 71.06
    ref_naive = 60.00
    ref_cutoff = [3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 0]

    cutoff = [findfirst(==(1), policy[t + 1, :]) - 1 for t in 0:T]   # smallest SELL state s*(t)

    @printf("pi match      : %s\n", all(isapprox.(pivec, ref_pi; atol = 1e-3)))
    @printf(
        "|lambda_2| ok : %s  (%.4f vs %.4f)\n",
        isapprox(slem(asset), ref_lam2; atol = 1e-3),
        slem(asset),
        ref_lam2
    )
    @printf(
        "pi.R ok       : %s  (%.2f vs %.2f)\n",
        isapprox(dot(pivec, R), ref_piR; atol = 1e-2),
        dot(pivec, R),
        ref_piR
    )
    @printf(
        "V_0 ok        : %s\n",
        all(isapprox.(round.(V[1, :]; digits = 2), ref_V0; atol = 1e-2))
    )
    @printf(
        "optimal ok    : %s  (%.2f vs %.2f)\n",
        isapprox(optimal, ref_opt; atol = 1e-2),
        optimal,
        ref_opt
    )
    @printf(
        "naive ok      : %s  (%.2f vs %.2f)\n",
        isapprox(naive, ref_naive; atol = 1e-2),
        naive,
        ref_naive
    )
    @printf("cutoff ok     : %s  (%s)\n", cutoff == ref_cutoff, cutoff)
end

# ╔═╡ Cell order:
# ╟─980248fd-ee83-44f3-9784-592148b383a2
# ╟─8de70c43-145b-4bde-b0da-3fc3b9ffef71
# ╠═5e30e618-d1ca-4884-a5da-1613e17aa785
# ╠═7d876aa4-d163-410d-b870-ada9c1b0e4d8
# ╠═4cc7dfa0-f937-4b61-ac2e-99ad733c532e
# ╟─1b2293a9-15b1-4547-90ba-bcfb58d32c8b
# ╠═3bf39c65-4da1-4f17-a5ff-f048939de668
# ╟─59e6b88a-9d2d-4581-9069-f5632e6954dc
# ╠═36384f99-29ce-4051-97f3-e5265dba74ab
# ╟─8eea75fd-9705-429f-a33f-cb0486a2f198
# ╟─0154692c-c4e2-474d-8385-6b6e3fb7ea2a
# ╠═097ffedd-c828-4731-be05-1f978100c9e7
# ╠═807262fd-22c4-44a3-84e4-f5010e8ec043
# ╟─cb35f633-3a49-49a5-9b36-6b990f607fe8
# ╟─ca246cb7-c3d6-4235-8ea9-b189302e4fc9
# ╠═2d685b8c-9916-43a2-884d-3628f33a8976
# ╟─117c3012-8b58-4107-8d4e-51bbcd1906e6
# ╠═79d9497e-9df5-4fb4-86ee-6b82e6a6acad
# ╟─e34d34e6-7aef-4944-b6b1-93c99c4b60fe
# ╠═8d000d41-440c-43dc-b8d0-ac4e4db54fa4
# ╠═3c253ac7-0cc5-4066-9a1c-65c9893424da
# ╟─67b80280-47b7-45a2-848c-e8ac95e8f210
# ╟─d4928829-8194-4e0a-a2f1-d6ada59f284b
# ╠═41612f26-8f50-46bb-991f-e593993cf44d
