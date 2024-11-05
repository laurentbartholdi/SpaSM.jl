# SpaSM

This is a user interface, in Julia, to the [SpaSM](https://github.com/cbouilla/spasm) sparse direct solver mod `p`.

It exposes (or will expose :)) all the functions of the C interface, in the same syntax but without the leading `spasm_`. It's currently very much a work in progress.

*Note* `spasm` creates a Spasm matrix from the *transpose* of a sparse one. Indeed Spasm stores matrix in row format while Julia stores them in column format, so that the "fast" conversion between Spasm and Julia matrices requires transposition.

```julia
julia> using Spasm, SparseArrays

julia> m = sparse([1,1,2,2],[1,2,1,2],[1,2,3,6]);

julia> sm = CSR(m)
2×2 CSR matrix % 42013 with 4 (maximum 4) non-zeros

julia> k = kernel(sm)
julia> k = kernel(sm)
[echelonize] Start on 2 x 2 matrix with 4 nnz
[echelonize] round 0
[pivots] Faugère-Lachartre: 1 pivots found [0.0s]
[pivots] ``Faugère-Lachartre on columns'': 0 pivots found [0.0s]
[pivots] greedy alternating cycle-free search: 0 pivots found [0.0s]
[pivots] 1 pivots found
Schur complement is 1 x 1, estimated density : 0.00 (4 byte)
Schur complement: 1 * 2 [0 nz / density= 0.000], 0.0s
[echelonize] round 2
[pivots] Faugère-Lachartre: 0 pivots found [0.0s]
[pivots] ``Faugère-Lachartre on columns'': 0 pivots found [0.0s]
[pivots] greedy alternating cycle-free search: 0 pivots found [0.0s]
[pivots] 0 pivots found
[echelonize] not enough pivots found; stopping
[echelonize] finishing; density = 0.000; aspect ratio = 0.5
[echelonize/GPLU] processing matrix of dimension 1 x 2

[echelonize/GPLU] full rank reached

[echelonize] Done in 0.0s. Rank 1, 2 nz in basis
[kernel] start. U is 1 x 2 (2 nnz). Transposing U
kernel: 1/1, |K| = 2
[kernel] done in 0.0s. NNZ(K) = 2
1×2 Spasm matrix % 42013 with 2 (maximum 2) non-zeros

julia> sparse(k)
2×1 SparseMatrixCSC{Spasm.GFp{42013}, Int64} with 2 stored entries:
 3
 42012
```

[![Build Status](https://github.com/laurentbartholdi/Spasm.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/laurentbartholdi/Spasm.jl/actions/workflows/CI.yml?query=branch%3Amain)
