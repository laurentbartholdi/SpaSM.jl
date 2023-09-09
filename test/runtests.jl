using Spasm, SparseArrays, LinearAlgebra, Test

m = sparse([1,1,3,3],[1,2,3,4],[1,2,3,4])

@testset "Construction of spasm matrix" begin
    sm = spasm(m)
    @test sparse(sm) == Spasm.GFp{42013}.(m)
end

@testset "Transpose" begin
    sm = spasm(m)
    @test sparse(transpose(transpose(sm))) == sparse(sm)
end

@testset "Kernel" begin
    sm = spasm(m)
    
    k = kernel(sm)
    @test sparse(k) == sparse([2],[1],Spasm.GFp{42013}[42012],3,1)

    @test sparse(kernel(transpose(sm))) == sparse([1,2,3,4],[1,1,2,2],Spasm.GFp{42013}[2,42012,28010,42012])
end
