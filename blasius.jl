using OrdinaryDiffEq, Roots
using DifferentialEquations
using LinearAlgebra, SparseArrays
using Plots, Printf

# ===========================================================
# 0) Parameters and nonuniform grid (dense near 0)
# ===========================================================
Xmax = 50.0
N    = 400
β    = 6.0

ξ = range(0.0, 1.0, length=N)
# Dense near 0: exponential mapping (dx/dξ grows with ξ)
x = Xmax .* (exp.(β .* ξ) .- 1.0) ./ (exp(β) - 1.0)

# ===========================================================
# 1) Blasius variant: f‴ + f f″ = 0  as first-order system
#    y1=f, y2=f′, y3=f″
# ===========================================================
function rhs!(du, u, p, t)
    f, fp, fpp = u
    du[1] = fp
    du[2] = fpp
    du[3] = -f * fpp
end

# ===========================================================
# 2) Robust shooting: early termination event + adaptive bracketing + Brent
# ===========================================================
function shoot_residual(a; X=Xmax)
    u0 = [0.0, 0.0, a]
    
    # Early terminate when f′ reaches 1
    condition(u, t, integrator) = u[2] - 1.0
    affect!(integrator) = terminate!(integrator)
    cb = ContinuousCallback(condition, affect!; rootfind=true)

    prob = ODEProblem(rhs!, u0, (0.0, X))

    # Nonstiff first; fallback to stiff if needed
    sol = solve(prob, Vern9(); abstol=1e-11, reltol=1e-9,
                save_everystep=false, callback=cb, maxiters=2_000_000)
    if sol.retcode != ReturnCode.Success
        sol = solve(prob, Rodas5P(); abstol=1e-11, reltol=1e-9,
                    save_everystep=false, callback=cb, maxiters=2_000_000)
    end

    return sol.u[end][2] - 1.0  # f′(end) - 1
end

function find_a_star(; X_init=Xmax, Xmax_max=200.0)
    # Physical solution has a>0; expand scan range and X if needed
    aL, aR = 1e-4, 5.0
    X_try = X_init

    for pass in 1:6
        avals = range(aL, aR; length=200)
        rvals = similar(collect(avals))
        for (k, a) in pairs(avals)
            r = shoot_residual(a; X=X_try)
            rvals[k] = r
            if isfinite(r) && abs(r) < 1e-10
                return a
            end
        end
        idx = findfirst(i -> isfinite(rvals[i]) && isfinite(rvals[i+1]) &&
                              rvals[i]*rvals[i+1] < 0, 1:length(avals)-1)
        if idx !== nothing
            return find_zero(a -> shoot_residual(a; X=X_try),
                             (avals[idx], avals[idx+1]), Brent();
                             atol=1e-12, rtol=1e-12, maxevals=300)
        end
        # Not bracketed: widen and increase terminal X
        aL /= 2; aR *= 2
        X_try = min(X_try*1.6, Xmax_max)
    end
    error("Failed to bracket root; increase Xmax or expand a-range.")
end

# ===========================================================
# 3) Shooting solution and sampling at custom grid
# ===========================================================
@info "Shooting to determine f″(0)"
a_star = find_a_star()
@info @sprintf("f″(0) ≈ %.10f  (shooting)", a_star)

u0_shoot = [0.0, 0.0, a_star]
prob     = ODEProblem(rhs!, u0_shoot, (0.0, Xmax))

sol = solve(prob, Vern9(); abstol=1e-11, reltol=1e-9, saveat=x, maxiters=2_000_000)
if sol.retcode != ReturnCode.Success
    sol = solve(prob, Rodas5P(); abstol=1e-11, reltol=1e-9, saveat=x, maxiters=2_000_000)
end

f_vals_shoot   = [sol(t; idxs=1) for t in x]
fp_vals_shoot  = [sol(t; idxs=2) for t in x]
fpp_vals_shoot = [sol(t; idxs=3) for t in x]

# ===========================================================
# 4) Finite-difference + Newton (nonuniform grid, correct closure)
#    Unknowns: f[1:n], fp[1:n], fpp[1:n]  total 3n
#    Equations:
#      - Interior i=2..n-1:
#          fp[i]  = D1(f)   (nonuniform central 3-point)
#          fpp[i] = D1(fp)
#          D1(fpp) + f[i]*fpp[i] = 0
#      - Boundary/closure (6 in total):
#          f[1]=0,  fp[1]=0,  fp[n]=1
#          fpp[1]  = D+ fp |_{1}
#          fpp[n]  = D- fp |_{n}
#          D- fpp |_{n} + f[n]*fpp[n] = 0
# ===========================================================
@inline function d1_coeffs(x, i)
    hm = x[i]   - x[i-1]
    hp = x[i+1] - x[i]
    c1m = -hp / (hm * (hm + hp))
    c1c =  (hp - hm) / (hm * hp)
    c1p =  hm / (hp * (hm + hp))
    return c1m, c1c, c1p
end

@inline idx_f(i)   = 3*(i-1) + 1
@inline idx_fp(i)  = 3*(i-1) + 2
@inline idx_fpp(i) = 3*(i-1) + 3

function build_R_J!(R, J, f, fp, fpp, x; fillJ::Bool)
    n = length(x)
    fill!(R, 0.0)
    if fillJ
        fill!(J.nzval, 0.0)
    end

    # Boundary: rows 1..3
    # 1) f(0)=0
    R[1] = f[1]
    if fillJ
        J[1, idx_f(1)] = 1.0
    end
    # 2) f′(0)=0
    R[2] = fp[1]
    if fillJ
        J[2, idx_fp(1)] = 1.0
    end
    # 3) f′(Xmax)=1
    R[3] = fp[n] - 1.0
    if fillJ
        J[3, idx_fp(n)] = 1.0
    end

    # Interior: i=2..n-1
    for i in 2:n-1
        r1 = 3*(i-1) + 1
        r2 = r1 + 1
        r3 = r2 + 1

        c1m, c1c, c1p = d1_coeffs(x, i)

        # 1) fp[i] - D1(f) = 0
        R[r1] = fp[i] - (c1m*f[i-1] + c1c*f[i] + c1p*f[i+1])
        if fillJ
            J[r1, idx_fp(i)] = 1.0
            J[r1, idx_f(i-1)] -= c1m
            J[r1, idx_f(i)]   -= c1c
            J[r1, idx_f(i+1)] -= c1p
        end

        # 2) fpp[i] - D1(fp) = 0
        R[r2] = fpp[i] - (c1m*fp[i-1] + c1c*fp[i] + c1p*fp[i+1])
        if fillJ
            J[r2, idx_fpp(i)] = 1.0
            J[r2, idx_fp(i-1)] -= c1m
            J[r2, idx_fp(i)]   -= c1c
            J[r2, idx_fp(i+1)] -= c1p
        end

        # 3) D1(fpp) + f[i]*fpp[i] = 0
        R[r3] = (c1m*fpp[i-1] + c1c*fpp[i] + c1p*fpp[i+1]) + f[i]*fpp[i]
        if fillJ
            J[r3, idx_fpp(i-1)] += c1m
            J[r3, idx_fpp(i)]   += c1c
            J[r3, idx_fpp(i+1)] += c1p
            J[r3, idx_f(i)]     += fpp[i]
            J[r3, idx_fpp(i)]   += f[i]
        end
    end

    # Tail closures: rows 3n-2, 3n-1, 3n
    h1    = x[2]   - x[1]
    hnm1  = x[end] - x[end-1]

    # 4) fpp[1] = D+ fp |_{1}
    r = 3*length(x) - 2
    R[r] = fpp[1] - (fp[2] - fp[1]) / h1
    if fillJ
        J[r, idx_fpp(1)] = 1.0
        J[r, idx_fp(2)] -= 1.0 / h1
        J[r, idx_fp(1)] += 1.0 / h1
    end

    # 5) fpp[n] = D- fp |_{n}
    r = 3*length(x) - 1
    R[r] = fpp[end] - (fp[end] - fp[end-1]) / hnm1
    if fillJ
        J[r, idx_fpp(length(x))] = 1.0
        J[r, idx_fp(length(x))] -= 1.0 / hnm1
        J[r, idx_fp(length(x)-1)] += 1.0 / hnm1
    end

    # 6) Right boundary ODE: D- fpp |_{n} + f[n]*fpp[n] = 0
    r = 3*length(x)
    R[r] = (fpp[end] - fpp[end-1]) / hnm1 + f[end]*fpp[end]
    if fillJ
        J[r, idx_fpp(length(x))]   += 1.0/hnm1 + f[end]
        J[r, idx_fpp(length(x)-1)] -= 1.0/hnm1
        J[r, idx_f(length(x))]     += fpp[end]
    end

    return nothing
end

function initial_guess(x; scale=25.0)
    n = length(x)
    fp  = tanh.(x ./ scale)
    fpp = (1.0/scale) .* (1 .- fp.^2)
    # Integrate to get f with trapezoidal accumulation; f(0)=0
    f = zeros(n)
    for i in 2:n
        h = x[i] - x[i-1]
        f[i] = f[i-1] + 0.5*h*(fp[i-1] + fp[i])
    end
    # Enforce initial boundary guesses
    f[1]    = 0.0
    fp[1]   = 0.0
    fp[end] = 1.0
    return f, fp, fpp
end

function solve_fd_newton(x; max_iter=60, tol=1e-8, λreg=1e-12)
    n = length(x)
    Ntot = 3*n

    f, fp, fpp = initial_guess(x; scale=25.0)

    R = zeros(Ntot)
    J = spzeros(Ntot, Ntot)

    build_R_J!(R, J, f, fp, fpp, x; fillJ=true)
    res0 = norm(R)
    @printf("Newton iter %2d, residual = %.3e\n", 0, res0)

    for iter in 1:max_iter
        Δ = (Matrix(J) + λreg*I) \ (-R)

        # Backtracking line search
        s = 1.0
        old_norm = norm(R)
        accept = false
        f_old, fp_old, fpp_old = copy(f), copy(fp), copy(fpp)

        for ls in 1:12
            for i in 1:n
                f[i]   = f_old[i]   + s * Δ[idx_f(i)]
                fp[i]  = fp_old[i]  + s * Δ[idx_fp(i)]
                fpp[i] = fpp_old[i] + s * Δ[idx_fpp(i)]
            end
            build_R_J!(R, J, f, fp, fpp, x; fillJ=false)
            new_norm = norm(R)
            if new_norm <= (1 - 1e-4*s) * old_norm
                accept = true
                break
            else
                s *= 0.5
            end
        end

        if !accept
            @warn "Line-search did not sufficiently decrease residual; using conservative step"
            s = 0.1
            for i in 1:n
                f[i]   = f_old[i]   + s * Δ[idx_f(i)]
                fp[i]  = fp_old[i]  + s * Δ[idx_fp(i)]
                fpp[i] = fpp_old[i] + s * Δ[idx_fpp(i)]
            end
            build_R_J!(R, J, f, fp, fpp, x; fillJ=false)
        end

        resn = norm(R)
        @printf("Newton iter %2d, residual = %.3e, step=%.2g\n", iter, resn, s)
        if resn < tol
            println("✅ FD-Newton converged")
            break
        end

        build_R_J!(R, J, f, fp, fpp, x; fillJ=true)
    end

    return f, fp, fpp
end

@info "Solving with FD-Newton"
f_fd, fp_fd, fpp_fd = solve_fd_newton(x)

# ===========================================================
# 5) Comparison and plots
# ===========================================================
println("\n==== Comparison ====")
@printf("Shooting:   f″(0) = %.10f, f′(∞) ≈ %.10f\n", a_star, fp_vals_shoot[end])
@printf("FD-Newton:  f″(0) = %.10f, f′(∞) ≈ %.10f\n", fpp_fd[1], fp_fd[end])

p1 = plot(x, f_vals_shoot, label="f(x) - Shooting", lw=2)
plot!(p1, x, fp_vals_shoot, label="f'(x) - Shooting")
plot!(p1, x, fpp_vals_shoot, label="f''(x) - Shooting")
plot!(p1, x, f_fd, label="f(x) - FD-Newton", linestyle=:dash, lw=2)
plot!(p1, x, fp_fd, label="f'(x) - FD-Newton", linestyle=:dash)
plot!(p1, x, fpp_fd, label="f''(x) - FD-Newton", linestyle=:dash)
xlabel!(p1, "x"); ylabel!(p1, "f, f', f''")
title!(p1, "Blasius: Shooting vs FD-Newton")

p2 = scatter(x, zeros(N), color=:red, label="Grid Points")
xlabel!(p2, "x"); ylabel!(p2, "Grid distribution")
title!(p2, "Nonuniform grid (dense near 0)")

plot(p1, p2, layout=(2,1), size=(900,700))
