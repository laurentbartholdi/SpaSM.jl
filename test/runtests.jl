using SpaSM, SparseArrays, LinearAlgebra, Test

m = sparse([1,1,3,3],[1,2,3,4],[1,2,3,4])

F = SpaSM.Field(42013)

@testset "Construction of spasm matrix" begin
    sm = CSR(m)
    @test sparse(sm) == SpaSM.ZZp{F}.(m)
end

@testset "Transpose" begin
    sm = CSR(m)
    @test sparse(transpose(transpose(sm))) == sparse(sm)
end

@testset "Kernel" begin
    sm = CSR(m)
    
    k = kernel(sm)
    @test sparse(k) == sparse([2],[1],SpaSM.ZZp{F}[42012],3,1)

    @test sparse(kernel(transpose(sm))) == sparse([1,2,3,4],[1,1,2,2],SpaSM.ZZp{F}[2,42012,28010,42012])
end
