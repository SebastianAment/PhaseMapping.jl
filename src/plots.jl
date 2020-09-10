# ternary plots
function checktern(a, b, c)
    all(x->x≈1, a + b + c) || throw("Factions do not sum to 1: $(a + b + c)")
end
terncoords(A::AbstractMatrix) = @views terncoords(A[1,:], A[2,:], A[3,:])
function terncoords(a::AbstractVector, b::AbstractVector,
                    c::AbstractVector = @.(1-a-b), clockwise::Val{true} = Val(true))
    checktern(a, b, c)
    y = c*sin(π/3)
    x = @. 1 - a - y*cot(π/3)
    return x, y
end

function ternary(A::AbstractMatrix, f::AbstractVector)
    x, y = terncoords(A)
    scatter(x, y, f)
    gui()
end

# using Makie
function ternary_surface(A::AbstractMatrix, f::AbstractVector)
    x, y = terncoords(A)
    surface(x, y, f)
    gui()
end

# mesh(
#            [(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)], color = [:red, :green, :blue],
#            shading = true
#        )

# counter-clockwise
# function terncoords(a::AbstractVector, b::AbstractVector,
#                     c::AbstractVector = @.(1-a-b), clockwise::Val{false})
#     checktern(a, b, c)
#     y = b*sin(π/3)
#     x = @. a + y*cot(π/3)
#     return x, y
# end
