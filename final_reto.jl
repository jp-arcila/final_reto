using LinearAlgebra, Statistics, Plots, CSV, DataFrames
using Optim, LsqFit


using CSV, DataFrames, LsqFit, Plots, Statistics

# Load and prepare data
df = CSV.read("temperaturas.csv", DataFrame)
rename!(df, [:"Tiempo (s)" => :t, :"Sensor 1" => :s1, :"Sensor 2" => :s2, 
            :"Sensor 3" => :s3, :"Sensor 4 (ambiente)" => :env])

# Find transition point (where cooling begins)
_, i_transition = findmax(df.s3)  # Using Sensor 3's peak as transition

# Newton's Law model
model(t, p) = p[1] .+ (p[2] - p[1]) .* exp.(-p[3] .* t)

function fit_sensor(data, t_range, T_env_est, label)
    # Initial parameters: [T_env, T_initial, k]
    p0 = [T_env_est, maximum(data[t_range]), 0.01]
    
    fit = curve_fit(model, df.t[t_range], data[t_range], p0)
    params = fit.param
    σ = stderror(fit)
    
    # Calculate R²
    y_pred = model(df.t[t_range], params)
    residuals = data[t_range] .- y_pred
    ss_res = sum(residuals.^2)
    ss_tot = sum((data[t_range] .- mean(data[t_range])).^2)
    r² = 1 - (ss_res/ss_tot)
    
    (; params, σ, r², y_pred)
end

# Analyze all sensors
results = Dict()
phases = ["heating", "cooling"]
ranges = [1:i_transition, i_transition:nrow(df)]

for (sensor, col) in [("Sensor 1", :s1), ("Sensor 2", :s2), ("Sensor 3", :s3)]
    results[sensor] = Dict()
    for (phase, r) in zip(phases, ranges)
        env_est = phase == "cooling" ? mean(df.env[r]) : df[first(r), col]
        res = fit_sensor(df[:, col], r, env_est, "$sensor $phase")
        results[sensor][phase] = res
    end
end

# Generate comparison plots
function create_plot(sensor, col)
    p = plot(xlabel="Time (s)", ylabel="Temperature (°C)", title="$sensor", legend=:bottomright)
    
    # Plot raw data
    scatter!(df.t, df[:, col], label="Experimental Data", markersize=2)
    
    # Plot fits
    for (phase, color) in zip(phases, [:red :blue])
        r = phase == "heating" ? (1:i_transition) : (i_transition:nrow(df))
        res = results[sensor][phase]
        plot!(df.t[r], res.y_pred, linewidth=2, color=color, 
            label="$(phase) fit (k=$(round(res.params[3],digits=4)), R²=$(round(res.r²,digits=3))")
    end
    
    # Add transition line
    vline!([df.t[i_transition]], linestyle=:dash, color=:black, label="Transition")
    
    p
end

# Create all plots
plots = [create_plot("Sensor $i", Symbol("s$i")) for i in 1:3]
plot(plots..., layout=(3,1), size=(800,1200))
savefig("sensors_comparison.png")

# Display results table
result_table = DataFrame(
    Sensor = String[],
    Phase = String[],
    T_env = Float64[],
    T_init = Float64[],
    k = Float64[],
    k_uncertainty = Float64[],
    R² = Float64[]
)

for sensor in keys(results)
    for phase in keys(results[sensor])
        res = results[sensor][phase]
        push!(result_table, (
            sensor,
            phase,
            res.params[1],
            res.params[2],
            res.params[3],
            res.σ[3],
            res.r²
        ))
    end
end

println("Fitting Results:")
show(result_table, allrows=true)