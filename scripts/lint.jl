# Run with:  julia --project=. scripts/lint.jl
# Requires JET and JuliaFormatter in your default Julia environment:
#   julia -e 'using Pkg; Pkg.add(["JET", "JuliaFormatter"])'

using JuliaFormatter
using JET

const ROOT = normpath(joinpath(@__DIR__, ".."))

println("== JuliaFormatter (check only) ==")
# overwrite=false => report-only; no files are modified. Returns true when
# everything is already formatted, false when at least one file would change.
ok = format(ROOT; overwrite = false)
if ok
    println("All files already formatted.")
else
    println("Some files are not formatted. Run `using JuliaFormatter; format(\"$(ROOT)\")` to fix.")
end

println()
println("== JET static analysis: test/runtests.jl ==")
try
    report_file(joinpath(ROOT, "test", "runtests.jl"))
catch err
    println("JET analysis skipped (tool error): ", err)
end

println()
println("Lint pass complete.")
