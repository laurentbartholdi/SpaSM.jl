module Spasm

using SparseArrays, Libdl, FileIO, Mmap

import Base: unsafe_convert
import SparseArrays: nnz

export spasm, kernel

const spasm_lib = "$(@__DIR__)" * "/../deps/spasm/src/libspasm." * Libdl.dlext

const prime₀ = 251 # the default prime to use

struct GFp{prime} v::Cuint end
Base.transpose(x::GFp) = x
GFp(x::Integer,prime = prime₀) = GFp{prime}(Cuint(mod(x,prime)))
GFp{prime}(x::Signed) where prime = GFp{prime}(Cuint(mod(x,prime)))
GFp(x::GFp) = x
#Base.convert(T::Type{GFp{prime}},x) where prime = T(x)
Base.convert(::Type{Int},x::GFp) = Int(x.v)
Base.zero(x::GFp) = typeof(x)(0)
Base.zero(T::Type{GFp{prime}}) where prime = T(0)
Base.one(x::GFp) = typeof(x)(1)
Base.one(T::Type{GFp{prime}}) where prime = T(1)
Base.show(io::IO,x::GFp) = print(io,"\e[1m",x.v,"\e[0m")
Base.Cuint(x::GFp) = x.v
Base.Int(x::GFp) = Int(x.v)
Base.Int32(x::GFp) = Int32(x.v)
Base.Int16(x::GFp{prime}) where prime = 2*Int(x.v)>prime ? Int16(x.v-prime) : Int16(x.v)
Base.:(*)(x::GFp{prime},y::GFp{prime}) where prime = GFp{prime}(mod(Int64(x.v)*y.v,prime))
Base.:(+)(x::GFp{prime},y::GFp{prime}) where prime = GFp{prime}(mod(Int64(x.v)+y.v,prime))
Base.:(-)(x::GFp{prime},y::GFp{prime}) where prime = GFp{prime}(mod(Int64(x.v)-y.v,prime))
Base.:(-)(x::GFp{prime}) where prime = GFp{prime}(mod(prime-x.v,prime))
Base.:(/)(x::GFp{prime},y::GFp{prime}) where prime = x*inv(v)
Base.:(*)(x::GFp,y::Bool) = y ? x : zero(x)
Base.:(*)(x::Bool,y::GFp) = x ? y : zero(y)
Base.:(*)(x::GFp,y::T) where T <: Integer = typeof(x)(x.v*y)
Base.:(*)(x::T,y::GFp) where T <: Integer = typeof(y)(x*y.v)

mutable struct spasm{prime}
    nzmax::Int64
    n::Int32 # number of rows
    m::Int32 # number of colums
    p::Ptr{Int64} # 0-based starts of rows
    j::Ptr{Int32} # 0-based column indices
    x::Ptr{GFp{prime}} # nonzeros
    prime__::Int32
end

function spasm(A::SparseMatrixCSC{Tv,Ti},prime = prime₀) where {Tv <: Integer,Ti} # actually create transposed matrix
    nzmax = length(A.nzval)
    m,n = size(A)
    spasmA = csr_alloc(n,m,nzmax,prime,true)
    nnz = 0
    spasmAx = convert(Ptr{Cuint},spasmA.x)
    for col=1:n
        unsafe_store!(spasmA.p,nnz,col)
        for i=A.colptr[col]:A.colptr[col+1]-1
#            v = GFp{prime}(A.nzval[i]) # causes huge memory allocation
            v = mod(A.nzval[i],prime)
            if !iszero(v)
                nnz += 1
                unsafe_store!(spasmA.j,A.rowval[i]-1,nnz)
#                unsafe_store!(spasmA.x,v,nnz)
                unsafe_store!(spasmAx,v,nnz)
            end
        end
    end
    unsafe_store!(spasmA.p,nnz,n+1)
    spasmA
end

function spasm(A::SparseMatrixCSC{GFp{prime},Ti}) where {prime,Ti} # actually create transposed matrix
    nzmax = length(A.nzval)
    m,n = size(A)
    spasmA = csr_alloc(n,m,nzmax,prime,true)
    nnz = 0
    for col=1:n
        unsafe_store!(spasmA.p,nnz,col)
        for i=A.colptr[col]:A.colptr[col+1]-1
            if !iszero(A.nzval[i])
                nnz += 1
                unsafe_store!(spasmA.j,A.rowval[i]-1,nnz)
                unsafe_store!(spasmA.x,A.nzval[i],nnz)
            end
        end
    end
    unsafe_store!(spasmA.p,nnz,n+1)
    spasmA
end

Base.show(io::IO, A::spasm{prime}) where prime = (@assert prime==A.prime__; print(io,A.n,"×",A.m," Spasm matrix % ",A.prime__," with ",nnz(A)," (maximum ",A.nzmax,") non-zeros"))

Base.size(A::spasm) = (A.n,A.m)
Base.size(A::spasm,i::Int) = (i==1 ? A.n : i==2 ? A.m : error("Invalid size index $i"))

function SparseArrays.sparse(A::spasm{prime}) where prime
    nnz = unsafe_load(A.p,A.n+1)
    colptr = [unsafe_load(A.p,i)+1 for i=1:A.n+1]
    rowval = [Int(unsafe_load(A.j,i))+1 for i=1:nnz]
    nzval = [unsafe_load(A.x,i) for i=1:nnz]
    for i=1:A.n
        range = colptr[i]:colptr[i+1]-1
        p = sortperm(view(rowval,range))
        permute!(view(rowval,range),p)
        permute!(view(nzval,range),p)
    end
    SparseMatrixCSC{GFp{prime},Int}(A.m,A.n,colptr,rowval,nzval)
end

spasm(x::Ptr{spasm{prime}}) where prime = (A = unsafe_load(x,1); @ccall free(x::Ptr{Cvoid})::Cvoid; finalizer(csr_free1,A); A)

permutation(x) = (x == nothing ? C_NULL : isa(x,Vector{Int32}) ? pointer(x) : error("$x should be nothing or a permutation{Cint}"))

mutable struct echelonize_opts
    enable_greedy_pivot_search::Bool
    enable_tall_and_skinny::Bool
    enable_dense::Bool
    enable_GPLU::Bool
    min_pivot_proportion::Float64
    max_round::Int32
    sparsity_threshold::Float64
    dense_block_size::Int
    low_rank_ratio::Float64
    tall_and_skinny_ratio::Float64
    low_rank_start_weight::Float64
end

# from spasm_util.c

nnz(A::spasm{prime}) where prime = @ccall spasm_lib."spasm_nnz"(pointer_from_objref(A)::Ptr{spasm{prime}})::Int64

csr_alloc(m,n,nzmax,prime,with_values) = spasm(convert(Ptr{spasm{prime}},@ccall spasm_lib."spasm_csr_alloc"(m::Int32,n::Int32,nzmax::Int64,prime::Int32,with_values::Int32)::Ptr{Cvoid}))

csr_realloc(A::spasm{prime},nzmax) where prime = @ccall spasm_lib."spasm_csr_realloc"(pointer_from_objref(A)::Ptr{spasm{prime}},nzmax::Int64)::Cvoid

csr_resize(A::spasm{prime},m,n) where prime = @ccall spasm_lib."spasm_csr_resize"(pointer_from_objref(A)::Ptr{spasm{prime}},m::Int32,n::Int32)::Cvoid

csr_free(A::spasm{prime}) where prime = @ccall spasm_lib."spasm_csr_free"(pointer_from_objref(A)::Ptr{spasm{prime}})::Cvoid

# usually, we should not directly call csr_free because the spasm structure itself belongs to Julia
csr_free1(A::spasm) = (@ccall free(A.p::Ptr{Cvoid})::Cvoid; @ccall free(A.j::Ptr{Cvoid})::Cvoid; @ccall free(A.x::Ptr{Cvoid})::Cvoid)

# declared, not defined
# identity(n,prime) = sparsm(@ccall spasm_lib."spasm_identity"(n::Int32,prime::Int32)::Ptr{spasm})

get_num_threads() = Int(@ccall spasm_lib."spasm_get_num_threads"()::Int32)
set_num_threads(n::Int) = @ccall spasm_lib."spasm_set_num_threads"(n::Cint)::Cvoid
get_thread_num() = Int(@ccall spasm_lib."spasm_get_thread_num"()::Int32)

# from spasm_transpose.c

Base.transpose(A::spasm{prime}) where prime = spasm(@ccall spasm_lib."spasm_transpose"(pointer_from_objref(A)::Ptr{spasm{prime}})::Ptr{spasm{prime}})

# from spasm_submatrix.c

submatrix(A::spasm{prime},r::UnitRange{Int},c::UnitRange{Int},with_values) where prime = spasm(@ccall spasm_lib."spasm_submatrix"(pointer_from_objref(A)::Ptr{spasm{prime}},r.start::Int32,r.stop::Int32,c.start::Int32,c.stop::Int32,with_value::Int32)::Ptr{spasm{prime}})

sorted_submatrix(A::spasm{prime},r::UnitRange{Int},c::UnitRange{Int},with_values) where prime = spasm(@ccall spasm_lib."sorted_spasm_submatrix"(pointer_from_objref(A)::Ptr{spasm{prime}},r.start::Int32,r.stop::Int32,c.start::Int32,c.stop::Int32,with_value::Int32)::Ptr{spasm{prime}})

rows_submatrix(A::spasm{prime},r::UnitRange{Int},with_values) where prime = spasm(@ccall spasm_lib."spasm_rows_submatrix"(pointer_from_objref(A)::Ptr{spasm{prime}},r.start::Int32,r.stop::Int32,with_value::Int32)::Ptr{spasm{prime}})

# from spasm_permutation.c

"""
# pvec(const int *p, const spasm_GFp * b, spasm_GFp * x, int n);
void spasm_ipvec(const int *p, const spasm_GFp * b, spasm_GFp * x, int n);

# pinv(p::Vector{Int32}) = int const *p, int n); # requires freeing
spasm *spasm_permute(const spasm * A, const int *p, const int *qinv, int with_values);
int *spasm_random_permutation(int n);
void spasm_range_pvec(int *x, int a, int b, int *p);
"""

# from spasm_GFp.c
Base.inv(a::GFp{prime}) where prime = @ccall spasm_lib."spasm_GFp_inverse"(a::GFp{prime},prime::Int32)::GFp{prime}

"""
/* spasm_scatter.c */
void spasm_scatter(const int *Aj, const spasm_GFp * Ax, int from, int to, spasm_GFp beta, spasm_GFp * x, int prime);

/* spasm_reach.c */
int spasm_dfs(int i, const spasm * G, int top, int *xi, int *pstack, int *marks, const int *pinv);
int spasm_reach(const spasm * G, const spasm * B, int k, int l, int *xi, const int *pinv);

/* spasm_gaxpy.c */
void spasm_gaxpy(const spasm * A, const spasm_GFp * x, spasm_GFp * y);
int spasm_sparse_vector_matrix_prod(const spasm * M, const spasm_GFp * x, const int *xi, int xnz, spasm_GFp * y, int *yi)
;

"""

# from spasm_triangular.c

is_upper_triangular(A::spasm{prime}) where prime = Bool(@ccall spasm_lib."spasm_is_upper_triangular"(pointer_from_objref(A)::Ptr{spasm{prime}})::Int32)

is_lower_triangular(A::spasm{prime}) where prime = Bool(@ccall spasm_lib."spasm_is_lower_triangular"(pointer_from_objref(A)::Ptr{spasm{prime}})::Int32)

# solve x . L = b where x and b are dense. Trashes b.
# p[j] == i indicates if the "diagonal" entry on column j is on row i.
dense_back_solve(L::spasm{prime}, b::Vector{GFp}, p = nothing, x::Vector{GFp} = Vector{GFp}(undef,L.n)) where prime = (@ccall spasm_lib."spasm_dense_back_solve"(pointer_from_objref(L)::Ptr{spasm{prime}}, pointer(b)::Ptr{Int32}, pointer(x)::Ptr{GFp}, permutation(p)::Ptr{Int32})::Cvoid; x)

# solve x . U = b where x and b are dense. Trashes b.
# q[i] indicates the column on which the i-th row pivot is.
dense_forward_solve(U::spasm{prime}, b::Vector{GFp}, q = nothing, x::Vector{GFp} = Vector{GFp}(undef,L.n)) where prime = (@ccall spasm_lib."spasm_dense_forward_solve"(pointer_from_objref(U)::Ptr{spasm{prime}}, pointer(b)::Ptr{Int32}, pointer(x)::Ptr{GFp}, permutation(q)::Ptr{Int32})::Cvoid; x)

sparse_backward_solve(L::spasm{prime}, B::spasm{prime}, k, pinv, r_bound, xi::Vector{Int32} = zeros(Int32,L.m), x::Vector{GFp} = Vector{GFp}(undef,L.m)) where prime = (top = @ccall spasm_lib."spasm_sparse_backward_solve"(pointer_from_objref(L)::Ptr{spasm{prime}}, pointer_from_objref(B)::Ptr{Int32}, k::Int32, pointer(xi)::Ptr{Int32}, pointer(x)::Ptr{GFp}, permutation(pinv)::Ptr{Int32}, r_bound::Int32)::Int32; (view(xi,top:L.m),x))

# solve x * U = B[k], where U is (permuted) upper triangular.
sparse_forward_solve(U::spasm{prime}, B::spasm{prime}, k, pinv = nothing, xi::Vector{Int32} = zeros(Int32,U.m), x::Vector{GFp} = Vector{GFp}(undef,U.m)) where prime = (top = @ccall spasm_lib."spasm_sparse_forward_solve"(pointer_from_objref(U)::Ptr{spasm{prime}}, pointer_from_objref(B)::Ptr{Int32}, k::Int32, pointer(xi)::Ptr{Int32}, pointer(x)::Ptr{GFp}, permutation(pinv)::Ptr{Int32})::Int32; (view(xi,top:U.m),x))

"""
/* spasm_lu.c */
spasm_lu *spasm_PLUQ(const spasm * A, const int *row_permutation, int keep_L);
spasm_lu *spasm_LU(const spasm * A, const int *row_permutation, int keep_L);
void spasm_free_LU(spasm_lu * X);
int spasm_find_pivot(int *xi, spasm_GFp * x, int top, spasm * U, spasm * L, int *unz_ptr, int *lnz_ptr, int i, int *deff_
ptr, int *qinv, int *p, int n);
void spasm_eliminate_sparse_pivots(const spasm * A, const int npiv, const int *p, spasm_GFp *x);

/* spasm_schur.c */
void spasm_make_pivots_unitary(spasm *A, const int *p, const int npiv);
void spasm_stack_nonpivotal_columns(spasm *A, int *qinv);
spasm *spasm_schur(spasm * A, int *p, int npiv, double est_density, int keep_L, int *p_out);
int spasm_schur_rank(spasm * A, const int *p, const int *qinv, const int npiv);
double spasm_schur_probe_density(spasm * A, const int *p, const int *qinv, const int npiv, const int R);

/* spasm_dense_lu.c */
spasm_dense_lu *spasm_dense_LU_alloc(int m, int prime);
void spasm_dense_LU_free(spasm_dense_lu * A);
int spasm_dense_LU_process(spasm_dense_lu *A, spasm_GFp *y);

/* spasm_solutions.c */
int spasm_PLUQ_solve(spasm * A, const spasm_GFp * b, spasm_GFp * x);
int spasm_LU_solve(spasm * A, const spasm_GFp * b, spasm_GFp * x);

/* spasm_pivots.c */
int spasm_find_pivots(spasm * A, int *p, int *qinv);
spasm * spasm_permute_pivots(const spasm *A, const int *p, int *qinv, int npiv);

/* spasm_matching.c */
int spasm_maximum_matching(const spasm * A, int *jmatch, int *imatch);
int *spasm_permute_row_matching(int n, const int *jmatch, const int *p, const int *qinv);
int *spasm_permute_column_matching(int m, const int *imatch, const int *pinv, const int *q);
int *spasm_submatching(const int *match, int a, int b, int c, int d);
int spasm_structural_rank(const spasm * A);

/* spasm_dm.c */
spasm_dm *spasm_dulmage_mendelsohn(const spasm * A);

/* spasm_scc.c */
spasm_dm *spasm_strongly_connected_components(const spasm * A);

/* spasm_ffpack.cpp */

void spasm_ffpack_setzero(int prime, int n, int m, double *A, int ldA);
int spasm_ffpack_echelonize(int prime, int n, int m, double *A, int ldA, size_t *qinv);
"""

# spasm_echelonize
echelonize_opts(opts = echelonize_opts((0 for _=1:11)...)) = (@ccall spasm_lib."spasm_echelonize_init_opts"(pointer_from_objref(opts)::Ptr{echelonize_opts})::Cvoid; opts)

function parse_echelonize_opts(kwargs)
    keys(kwargs)==(:echelonize_opts,) && return kwargs.echelonize_opts::echelonize_opts
    
    opts = echelonize_opts()
    for nv = kwargs
        setproperty!(opts,nv...)
    end
    opts
end


echelonize(A::spasm{prime}, Uqinv = Vector{Int32}(undef,A.m); kwargs...) where prime = (U = spasm(@ccall spasm_lib."spasm_echelonize"(pointer_from_objref(A)::Ptr{spasm{prime}},pointer(Uqinv)::Ptr{Int32},pointer_from_objref(parse_echelonize_opts(kwargs))::Ptr{echelonize_opts})::Ptr{spasm{prime}}); (U,Uqinv))

# spasm_rref.c
rref(U::spasm{prime},Uqinv = nothing, Rqinv = Vector{Int32}(undef,U.m)) where prime = (R = spasm(@ccall spasm_lib."spasm_rref"(pointer_from_objref(U)::Ptr{spasm{prime}},permutation(Uqinv)::Ptr{Int32},Rqinv::Ptr{Int32})::Ptr{spasm{prime}}); (R,Rqinv))

# from spasm_kernel.c
kernel(U::spasm{prime},column_permutation) where prime = spasm(@ccall spasm_lib."spasm_kernel"(pointer_from_objref(U)::Ptr{spasm{prime}},pointer(column_permutation)::Ptr{Int32})::Ptr{spasm{prime}})

kernel_from_rref(R::spasm{prime},column_permutation = nothing) where prime = spasm(@ccall spasm_lib."spasm_kernel"(pointer_from_objref(R)::Ptr{spasm{prime}},permutation(column_permutation)::Ptr{Int32})::Ptr{spasm{prime}})

# helpers

function load(f::File{format"SMS"}; kwargs...)
    open(f) do s; load(s; kwargs...) end
end

function load(s::Stream{format"SMS"}; kwargs...)
    load_sms(String(mmap(s.io)); kwargs...)
end
    
function save(f::File{format"SMS"}, A::SparseMatrixCSC; kwargs...)
    open(f, "w") do s; save(s, A; kwargs...) end
end

function save(s::Stream{format"SMS"}, A::SparseMatrixCSC; transpose=false)
    pb = PipeBuffer()
    write(pb,string(size(A,1))," ",string(size(A,2))," M\n")
    for (i,j,v) = zip(findnz(A)...)
        if transpose
            i,j = j,i
        end
        write(pb,string(i)," ",string(j)," ",string(v),"\n")
        bytesavailable(pb) > 65536 && write(s,take!(pb))
    end
    write(pb,"0 0 0\n")
    write(s,take!(pb))
end

function read_Int(str,pos) # much faster than parse
    v = 0
    neg = false
    while !isdigit(str[pos])
        if str[pos]=='-'
            neg = !neg
        end
        pos += 1
    end
    while true
        if isdigit(str[pos])
            v = 10v + Int(str[pos]-'0')
        else
            return (neg ? -v : v,pos+1)
        end
        pos += 1
    end
end

function load_sms(str::AbstractString; transpose=false, T=Int32)
    pos = 1
    (m,pos) = read_Int(str,pos)
    (n,pos) = read_Int(str,pos)
    # silently skip over the "M"
    Is = Int[]
    Js = Int[]
    Vs = T[]
    while true
        (i,pos) = read_Int(str,pos)
        (j,pos) = read_Int(str,pos)
        (v,pos) = read_Int(str,pos)
        if i == 0
            return transpose ? sparse(Js,Is,Vs,n,m) : sparse(Is,Js,Vs,m,n)
        end
        push!(Is,i)
        push!(Js,j)
        push!(Vs,v)
    end
end

const spasm_kernel_app = "$(@__DIR__)" * "/../deps/spasm/tools/kernel"

function kernel_sms(A, K, qinv = "/dev/null"; modulus = prime₀, dense_block_size = 280_000, left = false, enable_greedy_pivot_search = true, enable_tall_and_skinny = true, low_rank_start_weight = nothing)
    run(pipeline(`$spasm_kernel_app --modulus $modulus --dense-block-size $dense_block_size $(enable_greedy_pivot_search ? "" : "--no-greedy-pivot-search") $(enable_tall_and_skinny ? "" : "--no-low-rank-mode") $(low_rank_start_weight == nothing ? "" : "--low-rank-start-weight $low_rank_start_weight") --qinv-file $qinv`,stdin=A,stdout=K))
end

# one-stop shop for kernel computation
kernel(A::spasm{prime}; kwargs...) where prime = kernel(echelonize(A; kwargs...)...)

rank(A::spasm{prime}; kwargs...) where prime = count(≥(0),echelonize(A; kwargs...)[2])

end
