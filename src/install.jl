using Pkg

ssh = false
if ssh
    git = "git@github.com:SebastianAment/"
else
    git = "https://github.com/SebastianAment/"
end

add(x) = Pkg.add(Pkg.PackageSpec(url = git * x * ".git"))

add("LazyInverse.jl")
add("LinearAlgebraExtensions.jl")
add("WoodburyIdentity.jl")
add("KroneckerProducts.jl")
add("OptimizationAlgorithms.jl")
add("CovarianceFunctions.jl")
add("CompressedSensing.jl")
add("BackgroundSubtraction.jl")
add("PhaseMapping.jl")
