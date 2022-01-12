using Pkg

ssh = false
if ssh
    git = "git@github.com:SebastianAment/"
else
    git = "https://github.com/SebastianAment/"
end

add(x) = Pkg.add(Pkg.PackageSpec(url = git * x * ".git"))

add("OptimizationAlgorithms.jl")
add("BackgroundSubtraction.jl")
add("PhaseMapping.jl")
