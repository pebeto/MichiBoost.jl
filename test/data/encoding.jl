using MichiBoost: MichiBoost
using Test

@testset "encode_categorical on empty input" begin
    encoder = MichiBoost.OrderedTargetEncoder(0.5, 1.0, Dict{UInt32,Tuple{Float64,Int}}[])
    result = MichiBoost.encode_categorical(encoder, Vector{UInt32}[])
    @test size(result) == (0, 0)
end
