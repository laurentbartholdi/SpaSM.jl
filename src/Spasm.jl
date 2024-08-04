module Spasm

using SparseArrays, LinearAlgebra, StructArrays, Libdl, FileIO, Mmap, Images, Random, DataStructures

import SparseArrays: nnz
import LinearAlgebra: rank

export kernel, CSR, echelonize, Block, solve, gesv, sparse_triangular_solve, ZZp, Field, findnzs,
    rank, I, axpy!, xapy!, # from LinearAlgebra
    sprand, nnz, spzeros, findnz, # from SparseArrays
    load, save # from FileIO

#const spasm_lib = "$(@__DIR__)" * "/../deps/spasm/src/libspasm-asan." * Libdl.dlext
const spasm_lib = "$(@__DIR__)" * "/../deps/spasm/src/libspasm." * Libdl.dlext

const prime₀ = 42013 # the default prime to use

function _logtrue(s::Cstring)
    s = unsafe_string(s)
    msg = Base.text_colors[:yellow]*Base.text_colors[:bold]*"SPASM: "*Base.text_colors[:normal]
    if s[1]=='\r'
        s = "\r"*msg*s[2:end]
    elseif s[1]=='\n' && length(s)≠1
        s = "\n"*msg*s[2:end]
    else
        s = msg*s
    end
    print(s)
    Cint(0)
end

_logfalse(s::Cstring) = Cint(0)

function log(l = nothing)
    _logcallback = cglobal((:logcallback,Spasm.spasm_lib),Ptr{Nothing})

    if l==nothing
        unsafe_store!(_logcallback,C_NULL)
    elseif l==true
        unsafe_store!(_logcallback,@cfunction(_logtrue,Cint,(Cstring,)))
    elseif l==false
        unsafe_store!(_logcallback,@cfunction(_logfalse,Cint,(Cstring,)))
    else
        unsafe_store!(_logcallback,@cfunction($l,Cint,(Cstring,)))
    end
end

################################################################
# field arithmetic (mainly from spasm_ZZp.c)

struct Field
    p::Int
    halfp::Int
    mhalfp::Int
    dinvp::Float64
end

const SUBSCRIPT_DICT = Dict('0'=>'₀','1'=>'₁','2'=>'₂','3'=>'₃','4'=>'₄',
                            '5'=>'₅','6'=>'₆','7'=>'₇','8'=>'₈','9'=>'₉','-'=>'₋')
subscript_string(i) = prod(SUBSCRIPT_DICT[c] for c∈string(i))

Base.show(io::IO, f::Field) = print(io,"𝔽",subscript_string(f.p))

# cannot do that, because pointer_from_objref requires a mutable type.
#Field(p) = (f = Field(0,0,0,0.0); @ccall spasm_lib.spasm_field_init(p::Int, pointer_from_objref(f)::Ptr{Field})::Cvoid; f)

"""
    Field(p::Int)

Construct in Spasm the finite field 𝔽ₚ = ℤ/pℤ.
To construct its elements see `ZZp`.
"""
function Field(p)
    @assert 2 < p ≤ 0xfffffffb
    Field(p, p ÷ 2, p ÷ 2 - p + 1, 1. / p)
end
const F₀ = Field(prime₀)

struct ZZp{F} <: Number
    v::Int32
end

function _normalize(F::Field, x)
    if x < F.mhalfp x += F.p
    elseif x > F.halfp x -= F.p
    end
    ZZp{F}(Int32(x))
end

"""
    ZZp(F::Field, x::T) where {T <: Integer}
    ZZp(p::Int = $prime₀, x::T)

Construct in Spasm an element of the finite field 𝔽ₚ
"""
ZZp(F::Field, x::T) where {T <: Integer} = _normalize(F,mod(x,F.p))
ZZp(prime::Int, x::T) where {T <: Integer} = _normalize(Field(prime),mod(x,prime))
ZZp(x) = ZZp(F₀, x)
(F::Field)(x::T) where {T <: Integer} = _normalize(F,mod(x,F.p))
Base.convert(::Type{ZZp{F}},x::T) where {F, T <: Integer} = ZZp(F,x)
ZZp{F}(x::Spasm.ZZp{F}) where F = x
ZZp(x::ZZp{F₀}) = ZZp{F₀}(x)
ZZp(x::ZZp{F}) where F = error("Trying to convert from Field($(F.p)) to Field($prime₀)")

Base.Int8(x::ZZp) = Int8(x.v)
Base.Int16(x::ZZp) = Int16(x.v)
Base.Int32(x::ZZp) = Int32(x.v)
Base.Int64(x::ZZp) = Int64(x.v)

Base.UInt8(x::ZZp{F}) where F = UInt8(x.v < 0 ? x.v+F.p : x.v)
Base.UInt16(x::ZZp{F}) where F = UInt16(x.v < 0 ? x.v+F.p : x.v)
Base.UInt32(x::ZZp{F}) where F = UInt32(x.v < 0 ? x.v+F.p : x.v)
Base.UInt64(x::ZZp{F}) where F = UInt64(x.v < 0 ? x.v+F.p : x.v)

ZZp(F, x::Rational{T}) where {T <: Integer} = ZZp(F,x.num) * inv(ZZp(F,x.den))

Base.:(==)(x::ZZp{F}, y::ZZp{F}) where F = x.v == y.v

Base.hash(x::ZZp{F}, h::UInt) where F = hash(x.v, hash(F.p, h))

Random.rand(rng::AbstractRNG, ::Random.SamplerType{ZZp{F}}) where {F} = ZZp{F}(rand(rng,F.mhalfp:F.halfp))

################################################################
# spasm types

struct _CSR{F} # we need F as parameter to completely type x
    nzmax::Int64
    n::Int32 # number of rows
    m::Int32 # number of colums
    p::Ptr{Int64} # 0-based starts of rows
    j::Ptr{Int32} # 0-based column indices
    x::Ptr{ZZp{F}} # nonzeros
    field::Field
end

"""
    CSR{F} <: AbstractSparseMatrixCSC{ZZp{F},Int32}

Matrix type for Spasm's matrices in Compressed Sparse Row format.
The standard way of constructing CSR is via the Triplet format, or from
SparseArray's `SparseMatrixCSC` sparse matrices.
See also `zeros` and `sprand`.
"""
mutable struct CSR{F} <: AbstractSparseMatrix{ZZp{F},Int32}
    data::Ptr{_CSR{F}}
    function CSR(data::Ptr{_CSR{F}}; own=true) where F
        x = new{F}(data)
        own && finalizer(csr_free,x)
        x
    end
end
_get(x::CSR) = unsafe_load(x.data,1)

function Base.getproperty(N::CSR{F},s::Symbol) where F
    s===:data && return getfield(N,s)

    M = _get(N)
    if s===:p
        return unsafe_wrap(Vector{Int64},M.p,M.n+1)
    elseif s===:j
        return unsafe_wrap(Vector{Int32},M.j,M.nzmax)
    elseif s===:x
        return unsafe_wrap(Vector{ZZp{F}},M.x,M.nzmax)
    else
        return getfield(M,s)
    end
end

function Base.getindex(N::CSR{F},i::Integer,j::Integer) where F
    colptr = N.j
    for a=N.p[i]+1:N.p[i+1]
        if colptr[a] == j-1
            return N.x[a]
        end
    end
    return ZZp{F}(0)
end

# default, slow and ugly
function Base.getindex(X::CSR{F},i,j) where F
    subX = getindex(sparse(X),j,i)
    if isa(subX,SparseMatrixCSC)
        CSR(subX)
    elseif isa(subX,SparseVector)
        CSR(SparseMatrixCSC(subX))
    else
        subX
    end
end

# slow and ugly
Base.vcat(X1::CSR{F},X::CSR{F}) where F = CSR(hcat(sparse(X1),sparse(X)))
Base.hcat(X1::CSR{F},X::CSR{F}) where F = CSR(vcat(sparse(X1),sparse(X)))

Base.show(io::IO, A::CSR{F}) where F = (M = _get(A); @assert F==M.field; print(io,M.n,"×",M.m," CSR matrix % ",F.p," with ",nnz(A)," (maximum ",M.nzmax,") non-zeros"))

"""Run function f with as argument a Libc.FILE, and grabs what is written to the file into an IOBuffer."""
function grab_FILE(f::Function)
    bp = Ref{Ptr{UInt8}}(C_NULL)
    size = Ref(0)

    stream = @ccall open_memstream(pointer_from_objref(bp)::Ptr{Cvoid},pointer_from_objref(size)::Ptr{Int})::Libc.FILE
    f(stream)
    @ccall fclose(stream::Libc.FILE)::Cvoid
    IOBuffer(unsafe_wrap(Vector{UInt8},bp[],size[],own=true))
end

function Base.display(A::CSR{F}) where F
    if isdefined(Main, :IJulia) && Main.IJulia.inited
        n,m = size(A)
        maxsize = 500
        if max(m,n)>maxsize
            maxmn = max(m,n)
            n = n*maxsize ÷ maxmn
            m = m*maxsize ÷ maxmn
        end
        buf = grab_FILE() do f
            save_pnm(A,f,m,n,2)
        end
        display(load(File{format"PGM"}(buf)))
    else
        s = IOBuffer()
        display(TextDisplay(s),sparse(A,transpose=false))
        show(A)
        println("\n",join(readlines(seekstart(s))[2:end],"\n"))
    end
end

Base.size(A::CSR) = (M = _get(A); (M.n,M.m))
Base.size(A::CSR,i::Int) = size(A)[i]
Base.axes(A::CSR) = Base.OneTo.(size(A))
Base.axes(A::CSR,i::Int) = Base.OneTo(size(A,i))

struct _Triplet{F}
    nzmax::Int64
    nz::Int64
    n::Int32
    m::Int32
    i::Ptr{Int32}
    j::Ptr{Int32}
    x::Ptr{ZZp{F}}
    field::Field
end
mutable struct Triplet{F}
    data::Ptr{_Triplet{F}}
    function Triplet(data::Ptr{_Triplet{F}}) where F
        x = new{F}(data)
        finalizer(triplet_free,x)
        x
    end
end
_get(x::Triplet) = unsafe_load(x.data,1)

SparseArrays.nnz(A::Triplet) = (M = _get(A); M.nz)

Base.show(io::IO, A::Triplet{F}) where F = (M = _get(A); @assert F==M.field; print(io,M.n,"×",M.m," Triplet matrix % ",F.p," with ",nnz(A)," (maximum ",M.nzmax,") non-zeros"))

Base.size(A::Triplet) = (M = _get(A); (M.n,M.m))
Base.size(A::Triplet,i::Int) = size(A)[i]
Base.axes(A::Triplet) = Base.OneTo.(size(A))

mutable struct _LU{F}
    r::Int32
    complete::Int8 #bool
    L::Ptr{_CSR{F}}
    U::Ptr{_CSR{F}}
    qinv::Ptr{Int32}
    p::Ptr{Int32}
    Ltpm::Ptr{Triplet{F}}
end
mutable struct LU{F}
    data::Ptr{_LU{F}}
    function LU(data::Ptr{_LU{F}}) where F
        x = new{F}(data)
        finalizer(lu_free,x)
        x
    end
end
_get(x::LU) = unsafe_load(x.data,1)

Base.show(io::IO, N::LU{F}) where F = (M = _get(N); print(io,"LU: rank $(M.r), complete $(M.complete), L=$(M.L), U=$(M.U), qinv=$(M.qinv), p=$(M.p), Ltpm=$(M.Ltpm)"))

function Base.getproperty(N::LU,s::Symbol)
    s===:data && return getfield(N,s)
    
    M = _get(N)
    if s===:L
        M.L==C_NULL && error("M.L is null")
        return CSR(M.L,own=false)
    elseif s===:U
        M.U==C_NULL && error("M.U is null")
        return CSR(M.U,own=false)
    elseif s===:qinv
        M.qinv==C_NULL && error("M.qinv is null")
        M.U==C_NULL && error("M.U is null")
        return unsafe_wrap(Vector{Int32},M.qinv,unsafe_load(M.U,1).m)
    elseif s===:p
        M.p==C_NULL && error("M.p is null")
        M.U==C_NULL && error("M.U is null")
        return unsafe_wrap(Vector{Int32},M.p,unsafe_load(M.U,1).m)
    else
        return getfield(M,s)
    end
end
LinearAlgebra.rank(N::LU) = N.r

struct _DM
    p::Ptr{Int32} # size n, row permutation
    q::Ptr{Int32} # size m, column permutation
    r::Ptr{Int32} # size nb+1, block k is rows r[k] to r[k+1]-1 in A(p,q)
    c::Ptr{Int32} #  size nb+1, block k is cols s[k] to s[k+1]-1 in A(p,q)
    nb::Int32     # # of blocks in fine decomposition */
    rr::NTuple{5,Int32} # coarse row decomposition
    cc::NTuple{5,Int32} # coarse column decomposition
end

mutable struct DM # Dulmage-Mendelsohn
    data::_DM
    function DM(data::Ptr{_DM}; own=true)
        x = new(data)
        own && finalizer(dm_free,x)
    end
end

mutable struct EchelonizeOpts
    enable_greedy_pivot_search::Bool

    enable_tall_and_skinny::Bool
    enable_dense::Bool
    enable_GPLU::Bool

    L::Bool
    complete::Bool
    min_pivot_proportion::Float64
    max_round::Int32

    sparsity_threshold::Float64

    dense_block_size::Int
    low_rank_ratio::Float64
    tall_and_skinny_ratio::Float64
    low_rank_start_weight::Float64
end

mutable struct RankCertificate{F}
    r::Int32
    prime::Int64
    hash::NTuple{32,UInt8}
    i::Ptr{Int32}
    j::Ptr{Int32}
    x::Ptr{ZZp{F}}
    y::Ptr{ZZp{F}}
end

"""
typedef struct {
    u32 h[8];
    u32 Nl, Nh;
    u32 data[16];
    u32 num, md_len;
} spasm_sha256_ctx;

typedef struct {
        u32 block[11];   /* block[0:8] == H(matrix); block[8] = prime; block[9] = ctx, block[10] = seq */
        u32 hash[8];
        u32 prime;
        u32 mask;        /* 2^i - 1 where i is the smallest s.t. 2^i > prime */
        int counter;
        int i;
        spasm_field field;
} spasm_prng_ctx;

typedef enum {SPASM_DOUBLE, SPASM_FLOAT, SPASM_I64} spasm_datatype;

#define SPASM_IDENTITY_PERMUTATION NULL
#define SPASM_IGNORE NULL
#define SPASM_IGNORE_VALUES 0
"""

################################################################
# spasm_ZZp.c

Base.:(+)(a::ZZp{F}, b::ZZp{F}) where F = _normalize(F, Int(a.v) + Int(b.v))
Base.:(-)(a::ZZp{F}, b::ZZp{F}) where F = _normalize(F, Int(a.v) - Int(b.v))
Base.:(*)(a::ZZp{F}, b::ZZp{F}) where F = (q = Int(round(Float64(a.v) * Float64(b.v) * F.dinvp)); _normalize(F, Int(a.v) * Int(b.v) - q * F.p))
Base.inv(a::ZZp{F}) where F = _normalize(F, gcdx(a.v < 0 ? a.v + F.p : a.v, F.p)[2])
function axpy(a::ZZp{F}, x::ZZp{F}, y::ZZp{F}) where F
    q = Int(round((Float64(a.v) * Float64(x.v) + Float64(y.v)) * F.dinvp))
    _normalize(F, Int(a.v) * Int(x.v) + Int(y.v) - q * F.p)
end
Base.show(io::IO,x::ZZp{F}) where F = print(io,"\e[1m",x.v,"\e[0m")
Base.zero(x::ZZp) = typeof(x)(0)
Base.zero(T::Type{ZZp{F}}) where F = T(0)
Base.one(x::ZZp) = typeof(x)(1)
Base.one(T::Type{ZZp{F}}) where F = T(1)
Base.:(-)(x::ZZp{F}) where F = _normalize(F, -x.v)
Base.:(/)(x::ZZp{F},y::ZZp{F}) where F = x*inv(v)
Base.:(*)(x::ZZp,y::Bool) = y ? x : zero(x)
Base.:(*)(x::Bool,y::ZZp) = x ? y : zero(y)

Base.transpose(x::ZZp) = x
Base.adjoint(x::ZZp) = x
Base.copy(x::ZZp) = x

Base.:(*)(x::ZZp{F}, y::T) where {F, T <: Integer} = _normalize(F, Int(x.v)*y)
Base.:(*)(x::T, y::ZZp{F}) where {F, T <: Integer} = _normalize(F, x*Int(y.v))

################################################################
# sha256.c

# we don't expose these, they're already in the SHA package
"""
void spasm_SHA256_init(spasm_sha256_ctx *c);
void spasm_SHA256_update(spasm_sha256_ctx *c, const void *data, size_t len);
void spasm_SHA256_final(u8 *md, spasm_sha256_ctx *c);
"""

################################################################
# spasm_prng.c
"""
void spasm_prng_seed(const u8 *seed, i64 prime, u32 seq, spasm_prng_ctx *ctx);
void spasm_prng_seed_simple(i64 prime, u64 seed, u32 seq, spasm_prng_ctx *ctx);
u32 spasm_prng_u32(spasm_prng_ctx *ctx);
spasm_ZZp spasm_prng_ZZp(spasm_prng_ctx *ctx);
"""

################################################################
# spasm_util.c

wtime() = @ccall spasm_lib.spasm_wtime()::Float64

SparseArrays.nnz(A::CSR{F}) where F = @ccall spasm_lib.spasm_nnz(A.data::Ptr{_CSR{F}})::Int64

# we don't expose these, they're just the same as the Libc ones
"""
void *spasm_malloc(i64 size);
void *spasm_calloc(i64 count, i64 size);
void *spasm_realloc(void *ptr, i64 size);
"""

csr_alloc(m,n,nzmax,prime = prime₀,with_values = true) = CSR(convert(Ptr{_CSR{Field(prime)}},@ccall spasm_lib.spasm_csr_alloc(m::Int32,n::Int32,nzmax::Int64,prime::Int64,with_values::Bool)::Ptr{Cvoid}))

SparseArrays.spzeros(F::Field,m,n) = (M = csr_alloc(m,n,0,F.p); fill!(M.p,0); M)

SparseArrays.sprand(F::Field,m,n,density = 1.0) = CSR(sprand(ZZp{F},n,m,density))

csr_realloc(A::CSR{F},nzmax) where F = @ccall spasm_lib.spasm_csr_realloc(A.data::Ptr{_CSR{F}},nzmax::Int64)::Cvoid

csr_resize(A::CSR{F},m,n) where F = @ccall spasm_lib.spasm_csr_resize(A.data::Ptr{_CSR{F}},m::Int32,n::Int32)::Cvoid

csr_free(A::CSR{F}) where F = @ccall spasm_lib.spasm_csr_free(A.data::Ptr{_CSR{F}})::Cvoid

triplet_alloc(m,n,nzmax,prime = prime₀,with_values = true) = Triplet(convert(Ptr{_Triplet{Field(prime)}},@ccall spasm_lib.spasm_triplet_alloc(m::Int32,n::Int32,nzmax::Int64,prime::Int64,with_values::Bool)::Ptr{Cvoid}))

triplet_realloc(A::Triplet{F},nzmax) where F = @ccall spasm_lib.spasm_triplet_realloc(A.data::Ptr{_Triplet{F}},nzmax::Int64)::Cvoid

triplet_free(A::Triplet{F}) where F = @ccall spasm_lib.spasm_triplet_free(A.data::Ptr{_Triplet{F}})::Cvoid

dm_alloc(n,m) = DM(@ccall spasm_lib.spasm_dm_alloc(n::Int32, m::Int32)::Ptr{_DM})

dm_free(P::DM) = @ccall spasm_lib.spasm_dm_free(P.data::Ptr{_DM})::Cvoid

lu_free(N::LU{F}) where F = @ccall spasm_lib.spasm_lu_free(N.data::Ptr{_LU{F}})::Cvoid

# we won't expose this, it's just "string" in Julia
"""
void spasm_human_format(int64_t n, char *target);
"""

get_num_threads() = Int(@ccall spasm_lib.spasm_get_num_threads()::Int32)

# this doesn't seem to exist anymore
# set_num_threads(n::Int) = @ccall spasm_lib.spasm_set_num_threads(n::Cint)::Cvoid

get_thread_num() = Int(@ccall spasm_lib.spasm_get_thread_num()::Int32)

get_prime(A::CSR{F}) where F = (M = _get(A); @assert M.field == F; F.p)

################################################################
# spasm_triplet.c

function Base.push!(A::Triplet{F},ijx::Tuple{Int,Int,Int}) where F
    (i,j,x) = ijx
    @assert 1 ≤ i
    @assert 1 ≤ j
    @ccall spasm_lib.spasm_add_entry(A.data::Ptr{_Triplet{F}},(i-1)::Int32,(j-1)::Int32,x::Int64)::Cvoid
    A
end
Base.push!(A::Triplet{F},ijx::Tuple{Int,Int,ZZp{F}}) where F = push!(A,(ijx[1],ijx[2],ijx[3].v))

transpose!(A::Triplet{F}) where F = (@ccall spasm_lib.spasm_triplet_transpose(A.data::Ptr{_Triplet{F}})::Cvoid; A)

compress(A::Triplet{F}) where F = CSR(@ccall spasm_lib.spasm_compress(A.data::Ptr{_Triplet})::Ptr{_CSR{F}})

################################################################
# spasm_io.c

function triplet_load(f::Libc.FILE; prime = prime₀, get_hash = false)
    F = Field(prime)
    hash = get_hash ? Vector{UInt8}(undef,32) : nothing
    A = Triplet(convert(Ptr{_Triplet{F}},@ccall spasm_lib.spasm_triplet_load(f::Ptr{Libc.FILE},prime::Int64,(get_hash ? pointer(hash) : C_NULL)::Ptr{Cvoid})::Ptr{Cvoid}))
    get_hash ? (A,hash) : A
end
triplet_load(io::IO; kwargs...) = (f = Libc.FILE(io); M = triplet_load(f; kwargs...); close(f); M)

function fileio_load(f::File{format"SMS"}; csr = false, kwargs...)
    M = open(f) do s
        triplet_load(s.io; kwargs...)
    end

    csr ? compress(M) : M
end

triplet_save(A::Triplet{F}, f::Libc.FILE = Libc.FILE(RawFD(1), "w")) where F = @ccall spasm_lib.spasm_triplet_save(A.data::Ptr{_Triplet{F}},f.ptr::Ptr{Cvoid})::Cvoid
triplet_save(A::Triplet, io::IO) = (f = Libc.FILE(io); triplet_save(A,f); close(f); nothing)

function fileio_save(f::File{format"SMS"}, A::Triplet{F}) where F
    open(f,"w") do s
        triplet_save(A, s.io)
    end
end

csr_save(A::CSR{F}, f::Libc.FILE = Libc.FILE(RawFD(1), "w")) where F = @ccall spasm_lib.spasm_csr_save(A.data::Ptr{_CSR{F}},f.ptr::Ptr{Cvoid})::Cvoid
csr_save(A::CSR, io::IO) = (f = Libc.FILE(io); csr_save(A,f); close(f); nothing)    
function fileio_save(f::File{format"SMS"}, A::CSR{F}) where F
    open(f,"w") do s
        csr_save(A, s.io)
    end
end

#@ we ignore the spasm_dm field, for now
save_pnm(A::CSR{F}, f::Libc.FILE, x, y, mode, spasm_dm = nothing) where F = @ccall spasm_lib.spasm_save_pnm(A.data::Ptr{_CSR{F}},f.ptr::Ptr{Cvoid},x::Int32,y::Int32,mode::Int32,(spasm_dm == nothing ? C_NULL : spasm_dm.data)::Ptr{Cvoid})::Cvoid
function save_pnm(A::CSR, io::IO, x, y, mode, spasm_dm = nothing)
    intmode = Dict(:PBM => 1, :PGM => 2, :PPM => 3, 1 => 1, 2 => 2, 3 => 3)[mode]
    f = Libc.FILE(io);
    save_pnm(A, f, x, y, intmode, spasm_dm)
    close(f)
    nothing
end

#@ this does not work, save in format PNM is already registered for NetPBM
function fileio_save(f::Union{File{format"PBMBinary"},File{format"PGMBinary"},File{format"PPMBinary"}}, A::CSR{F}; x = 100, y = 100) where F
    mode = Dict(File{format"PBMBinary"} => 1,
                File{format"PGMBinary"} => 2,
                File{format"PPMBinary"} => 3)[typeof(f)]
    open(f,"w") do s
        save_pnm(A, s, x, y, mode)
    end
end

"""
Sample from the documentation:

struct WAVReader
    io::IO
    ownstream::Bool
end

function Base.read(reader::WAVReader, frames::Int)
    # read and decode audio samples from reader.io
end

function Base.close(reader::WAVReader)
    # do whatever cleanup the reader needs
    reader.ownstream && close(reader.io)
end

# FileIO has fallback functions that make these work using `do` syntax as well,
# and will automatically call `close` on the returned object.
loadstreaming(f::File{format"WAV"}) = WAVReader(open(f), true)
loadstreaming(s::Stream{format"WAV"}) = WAVReader(s, false)
If you choose to implement loadstreaming and savestreaming in your package, you can easily add save and load methods in the form of:

function save(q::Formatted{format"WAV"}, data, args...; kwargs...)
    savestreaming(q, args...; kwargs...) do stream
        write(stream, data)
    end
end

function load(q::Formatted{format"WAV"}, args...; kwargs...)
    loadstreaming(q, args...; kwargs...) do stream
        read(stream)
    end
end
"""
################################################################
# spasm_transpose.c

Base.transpose(A::CSR{F}) where F = CSR(@ccall spasm_lib.spasm_transpose(A.data::Ptr{_CSR{F}})::Ptr{_CSR{F}})

################################################################
# spasm_submatrix.c

function submatrix(A::CSR{F},r::AbstractUnitRange,c::AbstractUnitRange,with_values = true) where F
    @assert r ⊆ axes(A,1)
    @assert c ⊆ axes(A,2)
    CSR(@ccall spasm_lib.spasm_submatrix(A.data::Ptr{_CSR{F}},(minimum(r)-1)::Int32,maximum(r)::Int32,(minimum(c)-1)::Int32,maximum(c)::Int32,with_values::Bool)::Ptr{_CSR{F}})
end

Base.getindex(A::CSR,r::AbstractUnitRange,c::AbstractUnitRange) = submatrix(A,r,c)
Base.getindex(A::CSR,r::AbstractUnitRange,c::Colon) = submatrix(A,r,axes(A,2))
Base.getindex(A::CSR,r::Colon,c::AbstractUnitRange) = submatrix(A,axes(A,1),c)
Base.getindex(A::CSR,r::Colon,c::Colon) = submatrix(A,axes(A)...)

################################################################
# spasm_permutation.c
"""
void spasm_pvec(const int *p, const spasm_ZZp * b, spasm_ZZp * x, int n);
void spasm_ipvec(const int *p, const spasm_ZZp * b, spasm_ZZp * x, int n);
int *spasm_pinv(int const *p, int n);
struct spasm_csr *spasm_permute(const struct spasm_csr * A, const int *p, const int *qinv, int with_values);
int *spasm_random_permutation(int n);
void spasm_range_pvec(int *x, int a, int b, int *p);
"""

################################################################
# spasm_scatter.c

"""x += β*A[i]"""
scatter(A::CSR{F},i,β::ZZp{F},x::Vector{ZZp{F}}) where F = @ccall spasm_lib.spasm_scatter(A.data::Ptr{_CSR{F}},i::Int32,β::ZZp{F},pointer(x)::Ptr{ZZp{F}})::Cvoid

################################################################
# spasm_reach.c

# we don't expose these unless needed -- they're used internally
"""
int spasm_dfs(int i, const struct spasm_csr * G, int top, int *xi, int *pstack, int *marks, const int *pinv);
int spasm_reach(const struct spasm_csr * A, const struct spasm_csr * B, int k, int l, int *xj, const int *qinv);
"""

################################################################
# spasm_spmv.c

"""
    xApy(x::Vector{ZZp{F}}, A::CSR{F}, y::Vector{ZZp{F}})

Implements (dense vector) * (sparse CSR matrix)
y ← x⋅A + y
"""
function xapy!(x::Vector{ZZp{F}}, A::CSR{F}, y::Vector{ZZp{F}}) where F
    @assert length(x)==size(A,1)
    @assert length(y)==size(A,2)
    @ccall spasm_lib.spasm_xApy(pointer(x)::Ptr{Int32},A.data::Ptr{_CSR{F}},pointer(y)::Ptr{Int32})::Cvoid
end
Base.:(*)(x::Vector{ZZp{F}}, A::CSR{F}) where F = xapy!(x, A, zeros(ZZp{F},size(A,2)))

"""
    Axpy(A::CSR{F}, x::Vector{ZZp{F}}, y::Vector{ZZp{F}})

Implements (sparse CSR matrix) * (dense vector)
y ← A⋅x + y
"""
function LinearAlgebra.axpy!(A::CSR{F}, x::Vector{ZZp{F}}, y::Vector{ZZp{F}}) where F
    @assert length(x)==size(A,2)
    @assert length(y)==size(A,1)
    @ccall spasm_lib.spasm_Axpy(A.data::Ptr{_CSR{F}},pointer(x)::Ptr{Int32},pointer(y)::Ptr{Int32})::Cvoid
end
Base.:(*)(A::CSR{F}, x::Vector{ZZp{F}}) where F = axpy!(A, x, zeros(ZZp{F},size(A,1)))

################################################################
# spasm_triangular.c

"""
    dense_back_solve(L::CSR{F}, b::Vector{ZZp{F}}, x::Vector{ZZp{F}}, p::Vector{Int32})

Solve x⋅L = b with dense b and x.
x must have size n (#rows of L) and b must have size m (#cols of L)
b is destroyed
L is assumed to be (permuted) lower-triangular, with non-zero diagonal.

p[j] == i indicates if the "diagonal" entry on column j is on row i
"""
function dense_back_solve(L::CSR{F}, b::Vector{ZZp{F}}, x::Vector{ZZp{F}}, p::Vector{Int32}) where F
    @assert length(x)==size(L,1)
    @assert length(b)==size(L,2)==length(p)
    @ccall spasm_lib.spasm_dense_back_solve(L.data::Ptr{_CSR{F}}, pointer(b)::Ptr{Int32}, pointer(x)::Ptr{Int32}, Ptr{p}::Ptr{Int32})::Bool
end

"""
    dense_forward_solve(U::CSR{F}, b::Vector{ZZp{F}}, x::Vector{ZZp{F}}, q::Vector{Int32})

Solve x⋅U = b with dense x, b.
b is destroyed on output

U is (permuted) upper-triangular with unit diagonal.
q[i] == j means that the pivot on row i is on column j (this is the inverse of the usual qinv).
"""
function dense_forward_solve(U::CSR{F}, b::Vector{ZZp{F}}, x::Vector{ZZp{F}}, q::Vector{Int32}) where F
    @assert length(x)==size(U,1)==length(q)
    @assert length(b)==size(U,2)
    @ccall spasm_lib.spasm_dense_forward_solve(U.data::Ptr{_CSR{F}}, pointer(b)::Ptr{Int32}, pointer(x)::Ptr{Int32}, Ptr{q}::Ptr{Int32})::Bool
end

"""
    sparse_triangular_solve(U::CSR{F}, B::CSR{F}, k::Int, xj::Vector{Int32}, x::Vector{ZZp{F}}, qinv::Vector{Int32})

Solve x * U = B[k], where U is (permuted) triangular (either upper or lower).

x must have size m (#columns of U); it does not need to be initialized.
xj must be preallocated of size 3*m and zero-initialized (it remains OK)
qinv locates the pivots in U.

On output, the solution is scattered in x, and its pattern is given in xj[top:m].
The precise semantics is as follows. Define:
         x_a = { j in [0:m] : qinv[j] < 0 }
         x_b = { j in [0:m] : qinv[j] >= 0 }
Then x_b * U + x_a == B[k].  It follows that x * U == y has a solution iff x_a is empty.

top is the return value

This does not require the pivots to be the first entry of the row.
This requires that the pivots in U are all equal to 1.
"""
function sparse_triangular_solve(U::CSR{F}, B::CSR{F}, k::Int, xj::Vector{Int32}, x::Vector{ZZp{F}}, qinv::Vector{Int32}) where F
    m = size(U,2)
    @assert m==size(B,2)==length(qinv)
    @assert 0≤k<size(B,1)
    @assert all(iszero,xj)
    @assert length(xj)≥3m
    @assert length(x)≥m
    @ccall spasm_lib.spasm_sparse_triangular_solve(U.data::Ptr{_CSR{F}}, B.data::Ptr{_CSR{F}}, k::Int32, pointer(xj)::Ptr{Int32}, pointer(x)::Ptr{Int32}, pointer(qinv)::Ptr{Int32})::Int32
end

"""
    sparse_triangular_solve(LU::LU{F}, B::CSR{F})
    sparse_triangular_solve(U::CSR{F}, B::CSR{F}, qinv::Vector{Int32})

Solve `X`*`LU.U` == `B`, respectively `X`*`U` = `B` in sparse matrices.
The pivot positions of `U` are indicated in `qinv`.

Returns the solution `X` as a sparse matrix, or `nothing` if there is no solution.
"""
function sparse_triangular_solve(U::CSR{F}, B::CSR{F}, qinv::Vector{Int32}) where F
    m = size(U,2)
    @assert m==size(B,2)==length(qinv)
    Xt = spzeros(ZZp{F},size(U,1),size(B,1)) # transposed solution
    xj = Vector{Int32}(undef,3m)
    x = Vector{ZZp{F}}(undef,m)
    for k=1:size(B,1)
        fill!(xj,0)
        top = sparse_triangular_solve(U, B, k-1, xj, x, qinv)
        for i=top+1:m
            j = xj[i]+1
            iszero(x[j]) && continue
            qinv[j]<0 && return nothing
            push!(Xt.rowval,qinv[j]+1)
            push!(Xt.nzval,x[j])
        end
        Xt.colptr[k+1] = length(Xt.rowval)+1
    end
    CSR(Xt)
end

sparse_triangular_solve(LU::LU{F}, B::CSR{F}) where F = sparse_triangular_solve(LU.U, B, LU.qinv)
Base.:(/)(B::CSR{F}, LU::LU{F}) where F = sparse_triangular_solve(LU, B)

################################################################
# spasm_schur.c

"""
struct spasm_csr *spasm_schur(const struct spasm_csr *A, const int *p, int n, const struct spasm_lu *fact,
                   double est_density, struct spasm_triplet *L, const int *p_in, int *p_out);
double spasm_schur_estimate_density(const struct spasm_csr * A, const int *p, int n, const struct spasm_csr *U, const int
 *qinv, int R);
void spasm_schur_dense(const struct spasm_csr *A, const int *p, int n, const int *p_in,
        struct spasm_lu *fact, void *S, spasm_datatype datatype,int *q, int *p_out);
void spasm_schur_dense_randomized(const struct spasm_csr *A, const int *p, int n, const struct spasm_csr *U, const int *q
inv,
        void *S, spasm_datatype datatype, int *q, int N, int w);
"""

################################################################
# spasm_pivots.c

"""
int spasm_pivots_extract_structural(const struct spasm_csr *A, const int *p_in, struct spasm_lu *fact, int *p, struct ech
elonize_opts *opts);
"""

################################################################
# spasm_matching.c

"""
int spasm_maximum_matching(const struct spasm_csr *A, int *jmatch, int *imatch);
int *spasm_permute_row_matching(int n, const int *jmatch, const int *p, const int *qinv);
int *spasm_permute_column_matching(int m, const int *imatch, const int *pinv, const int *q);
int *spasm_submatching(const int *match, int a, int b, int c, int d);
int spasm_structural_rank(const struct spasm_csr *A);
"""

################################################################
# spasm_dm.c

dulmage_mendelsohn(A::CSR{F}) where F = DM(@ccall spasm_lib.spasm_dulmage_mendelsohn(A.data::Ptr{_CSR{F}})::Ptr{_DM})

################################################################
# spasm_scc.c

strongly_connected_components(A::CSR{F}) where F = DM(@ccall spasm_lib.spasm_strongly_connected_components(A.data::Ptr{_CSR{F}})::Ptr{_DM})

################################################################
# spasm_ffpack.cpp

"""
int spasm_ffpack_rref(i64 F, int n, int m, void *A, int ldA, spasm_datatype datatype, size_t *qinv);
int spasm_ffpack_LU(i64 F, int n, int m, void *A, int ldA, spasm_datatype datatype, size_t *p, size_t *qinv);
spasm_ZZp spasm_datatype_read(const void *A, size_t i, spasm_datatype datatype);
void spasm_datatype_write(void *A, size_t i, spasm_datatype datatype, spasm_ZZp value);
size_t spasm_datatype_size(spasm_datatype datatype);
spasm_datatype spasm_datatype_choose(i64 F);
const char * spasm_datatype_name(spasm_datatype datatype);
"""

################################################################
# spasm_echelonize

EchelonizeOpts(opts = EchelonizeOpts((0 for _=1:13)...)) = (@ccall spasm_lib.spasm_echelonize_init_opts(pointer_from_objref(opts)::Ptr{EchelonizeOpts})::Cvoid; opts)

function parse_echelonize_opts(opts = EchelonizeOpts(); kwargs...)
    for nv = kwargs
        setproperty!(opts,nv...)
    end
    opts
end

"""Capture stderr while executing f, if the optional argument is true.

Typical syntax is
julia> capture_stderr() do
           @info "Hi!"
            42
       end

(42, "┌ Info: Hi!\n└ @ Main REPL[1]:2\n")

This plays badly with multithreading, so is automatically disabled when `Threads.threadid() ≥ 2`
"""
function capture_stderr(f::Function, activate::Bool = true)
    activate &= Threads.threadid() == 1
    
    if activate
        original_stderr = stderr
        err_rd, err_wr = redirect_stderr()
        err_reader = @async read(err_rd, String)
    end
    
    result = f()

    if activate
        redirect_stderr(original_stderr)
        close(err_wr)
        message = fetch(err_reader)
    else
        message = ""
    end
    
    result, message
end

function echelonize(A::CSR{F}, opts = EchelonizeOpts(); verbose = false, kwargs...) where F
    isa(verbose,Bool) || (verbose = nnz(A) ≥ verbose)
    lu, _ = capture_stderr(!verbose) do
        LU(@ccall spasm_lib.spasm_echelonize(A.data::Ptr{_CSR{F}},pointer_from_objref(parse_echelonize_opts(opts; kwargs...))::Ptr{EchelonizeOpts})::Ptr{_LU{F}})
    end
    lu
end

################################################################
# spasm_rref.c

rref(fact::LU{F}, Rqinv) where F = CSR(@ccall spasm_lib.spasm_rref(fact.data::Ptr{_LU{F}}, pointer(Rqinv)::Ptr{Int32})::Ptr{_CSR{F}})

################################################################
# spasm_kernel.c

function kernel(fact::LU{F}; verbose = false) where F
    isa(verbose,Bool) || (verbose = nnz(fact.U) ≥ verbose)
    k, _ = capture_stderr(!verbose) do
        CSR(@ccall spasm_lib.spasm_kernel(fact.data::Ptr{_LU{F}})::Ptr{_CSR{F}})
    end
    k
end

kernel_from_rref(R::CSR{F}, qinv) where F = CSR(@ccall spasm_lib.spasm_kernel(fact.data::Ptr{_LU{F}}, qinv::Ptr{Int32})::Ptr{_CSR{F}})

################################################################
# spasm_solve.c

"""
    solve(fact::LU{F}, b::Vector{ZZp{F}}, x::Union{Vector{ZZp{F}},Nothing} = nothing)

Solve `x`*`A` == `b`, where `A` has already been echelonized as `fact`.
Returns solution `x` or `nothing` if no solution
"""
function solve(fact::LU{F}, b::Vector{ZZp{F}}, x::Union{Vector{ZZp{F}},Nothing} = nothing) where F
    fact.L # force it to be non-null
    @assert length(b)==fact.U.m
    if x==nothing
        x = zeros(ZZp{F},fact.U.n)
    else
        @assert length(x)==fact.U.n
    end
    ok = @ccall spasm_lib.spasm_solve(fact.data::Ptr{_LU{F}}, pointer(b)::Ptr{Int32}, pointer(x)::Ptr{Int32})::Bool
    ok ? x : nothing
end

"""
    gesv(fact::LU{F}, B::CSR{F})

Solve `X`*`A` == `B` where `A` has already been echelonized in `fact`.

Assumes that `fact` is a complete factorization, with `:U` and `:L` fields.
Returns (X, Vector{Bool} indicating which rows of `X` are valid solutions).
"""
function gesv(fact::LU{F}, B::CSR{F}; verbose = false) where F
    fact.L
    @assert size(B,2)==size(fact.U,2)
    ok = zeros(Bool,size(B,1))
    X, _ = capture_stderr(!verbose) do
	CSR(@ccall spasm_lib.spasm_gesv(fact.data::Ptr{_LU{F}}, B.data::Ptr{_CSR{F}}, pointer(ok)::Ptr{Bool})::Ptr{_CSR{F}})
    end
    (X,ok)
end

################################################################
# spasm_certificate.c

certificate_rank_create(A::CSR{F}, hash::Vector{UInt8}, fact::LU{F}) where F = @ccall spasm_lib.spasm_certificate_rank_create(A.data::Ptr{_CSR{F}},pointer(hash)::Ptr{UInt8},fact.data::Ptr{_LU{F}})::RankCertificate{F}

certificate_rank_verify(A::CSR{F}, hash::Vector{UInt8}, proof::RankCertificate{F}) where F = @ccall spasm_lib.spasm_certificate_rank_verify(A.data::Ptr{_CSR{F}}, pointer(hash)::Ptr{UInt8}, proof::RankCertificate{F})::Bool

rank_certificate_save(proof::RankCertificate{F},f::Libc.FILE) where F = @ccall spasm_lib.spasm_rank_certificate_save(proof::RankCertificate{F},f::Libc.FILE)::Cvoid

rank_certificate_load(f::Libc.FILE,proof::RankCertificate{F}) where F = @ccall spasm_lib.spasm_rank_certificate_load(f::Libc.FILE,proof::RankCertificate{F})::Bool

factorization_verify(A::CSR{F}, fact::LU{F}, seed::UInt64) where F = @ccall spasm_lib.spasm_factorization_verify(A.data::Ptr{_CSR{F}}, fact.data::Ptr{_LU{F}}, seed::UInt64)::Bool

################################################################
# more constructors

function CSR(A::SparseMatrixCSC{Tv,Ti}, prime = prime₀; transpose = true) where {Tv <: Number,Ti}
    nzmax = length(A.nzval)
    m,n = size(A)
    realA = csr_alloc(n,m,nzmax,prime,true)
    spasmA = _get(realA)
    nnz = 0
    spasmAx = convert(Ptr{Int32},spasmA.x)
    for col=1:n
        unsafe_store!(spasmA.p,nnz,col)
        for i=A.colptr[col]:A.colptr[col+1]-1
            v = A.nzval[i]
            if isa(v,Rational)
                v = v.num*gcdx(v.den < 0 ? v.den + prime : v.den, prime)[2]
            end
            v = mod(v,prime)
            if 2v>prime
                v -= prime
            end
            if !iszero(v)
                nnz += 1
                unsafe_store!(spasmA.j,A.rowval[i]-1,nnz)
                unsafe_store!(spasmAx,v,nnz)
            end
        end
    end
    unsafe_store!(spasmA.p,nnz,n+1)
    transpose ? realA : Base.transpose(realA)
end

function CSR(A::SparseMatrixCSC{ZZp{F},Ti}; transpose = true) where {F,Ti}
    nzmax = length(A.nzval)
    m,n = size(A)
    realA = csr_alloc(n,m,nzmax,F.p,true)
    spasmA = _get(realA)
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
    transpose ? realA : Base.transpose(realA)
end

CSR(a::UniformScaling{F},n) where F = CSR(sparse(a,n,n))
CSR(a::UniformScaling{ZZp{F}},n) where F = CSR(sparse(a.λ*I,n,n))
CSR(a::UniformScaling{T},n) where T <: Number = CSR(sparse(ZZp(a.λ)*I,n,n))

#@ the following are inefficient, we could work directly on the matrices without going through SparseMatrixCSC
Base.:(*)(a::UniformScaling{T}, b::CSR{F}) where {F,T <: Number} = CSR(ZZp{F}(a.λ)*sparse(b))
Base.:(*)(a::CSR{F}, b::UniformScaling{T}) where {F,T <: Number} = CSR(sparse(a)*ZZp{F}(b.λ))
Base.:(*)(a::ZZp{F}, b::CSR{F}) where F = CSR(a*sparse(b))
Base.:(*)(a::T, b::CSR{F}) where {F,T <: Number} = CSR(ZZp{F}(a)*sparse(b))
Base.:(*)(a::CSR{F}, b::ZZp{F}) where F = CSR(sparse(a)*b)
Base.:(*)(a::CSR{F}, b::T) where {F, T <: Number} = CSR(sparse(a)*ZZp{F}(b))
Base.:(*)(a::CSR{F}, b::CSR{F}) where F = CSR(sparse(b)*sparse(a))
Base.:(+)(a::CSR{F}, b::CSR{F}) where F = CSR(sparse(a)+sparse(b))
Base.:(-)(a::CSR{F}, b::CSR{F}) where F = CSR(sparse(a)-sparse(b))
Base.:(-)(a::CSR{F}) where F = CSR(-sparse(a))
Base.:(==)(a::CSR{F}, b::CSR{F}) where F = sparse(a) == sparse(b)
Base.hash(a::CSR{F}, h...) where F = hash(sparse(a), h...)

"""Sort two arrays in parallel, using `a` for comparison"""
parallelsort!(a,b) = sort!(StructArray((a,b)),lt=(x,y)->x[1]<y[1],alg=Base.Sort.QuickSort)

function SparseArrays.sparse(A::CSR{F}; transpose = true) where F
    A = _get(A)
    nnz = unsafe_load(A.p,A.n+1)
    colptr = [unsafe_load(A.p,i)+1 for i=1:A.n+1]
    rowval = [Int(unsafe_load(A.j,i))+1 for i=1:nnz]
    nzval = [unsafe_load(A.x,i) for i=1:nnz]
    for i=1:A.n
        range = colptr[i]:colptr[i+1]-1
        parallelsort!(view(rowval,range),view(nzval,range))
    end
    mat = SparseMatrixCSC{ZZp{F},Int}(A.m,A.n,colptr,rowval,nzval)
    transpose ? mat : Base.transpose(mat)
end

function fileio_save(f::File{format"SMS"}, A::SparseMatrixCSC; kwargs...)
    open(f, "w") do s; save(s, A; kwargs...) end
end

function fileio_save(s::Stream{format"SMS"}, A::SparseMatrixCSC; transpose=false)
    pb = PipeBuffer()
    write(pb,string(size(A,1))," ",string(size(A,2))," M\n")
    for (i,j,v) = zip(findnz(A)...)
        if transpose
            i,j = j,i
        end
        write(pb,string(i)," ",string(j)," ",string(Int(v)),"\n")
        bytesavailable(pb) > 2^20 && write(s,take!(pb))
    end
    write(pb,"0 0 0\n")
    write(s,take!(pb))
    nothing
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

function sms_to_sparse(f::AbstractString; transpose=false, T=Int32)
    open(f) do stream
        str = String(mmap(stream))
        
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
end

function SparseArrays.findnz(A::CSR{F}) where F
    numnz = nnz(A)
    p = A.p
    J = A.j .+ 1
    V = copy(A.x)
    I = Vector{Int}(undef, numnz)

    count = 1
    for row=1:size(A,1), k=A.p[row]+1:A.p[row+1]
        I[count] = row
        count += 1
    end

    return (I, J, V)
end

struct CSREnumerator{F}
   A::CSR{F}
end
Base.length(enum::CSREnumerator{F}) where F = nnz(enum.A)

"""Return a lightweight iterator over all non-zeros in matrix `A`, as triples `(i,j,v)`
"""
findnzs(A::CSR{F}) where F = CSREnumerator{F}(A)

function Base.iterate(enum::CSREnumerator{F}, state = (1,1)) where F
    row, pos = state
    row > enum.A.n && return nothing
    pos > enum.A.p[row+1] && return nothing
    ((row, enum.A.j[pos]+1, enum.A.x[pos]), (row+Int(enum.A.p[row+1]==pos),pos+1))
end

################################################################
# for testing, direct access to the compiled programs

const spasm_kernel_app = "$(@__DIR__)" * "/../deps/spasm/tools/kernel"
const spasm_rank_app = "$(@__DIR__)" * "/../deps/spasm/tools/rank"

function kernel_sms(A, K, qinv = "/dev/null"; modulus = F₀, dense_block_size = nothing, left = false, enable_greedy_pivot_search = true, enable_tall_and_skinny = true, low_rank_start_weight = nothing, num_threads = 1)
    errmsg = IOBuffer()
    try
	run(pipeline(addenv(`$spasm_kernel_app --modulus $modulus $(dense_block_size == nothing ? "" : "--dense-block-size $dense_block_size") $(enable_greedy_pivot_search ? "" : "--no-greedy-pivot-search") $(enable_tall_and_skinny ? "" : "--no-low-rank-mode") $(low_rank_start_weight == nothing ? "" : "--low-rank-start-weight $low_rank_start_weight") --qinv-file $qinv`,"OMP_NUM_THREADS"=>string(num_threads)),stdin=A,stdout=K,stderr=errmsg))
    catch
	@error String(readavailable(seekstart(errmsg)))
    end
end

function rank_sms(A; modulus = F₀, dense_block_size = nothing, left = false, enable_greedy_pivot_search = true, enable_tall_and_skinny = true, low_rank_start_weight = nothing, num_threads = 1)
    errmsg = IOBuffer()
    try
	run(pipeline(addenv(`$spasm_rank_app --modulus $modulus $(dense_block_size == nothing ? "" : "--dense-block-size $dense_block_size") $(enable_greedy_pivot_search ? "" : "--no-greedy-pivot-search") $(enable_tall_and_skinny ? "" : "--no-low-rank-mode") $(low_rank_start_weight == nothing ? "" : "--low-rank-start-weight $low_rank_start_weight")`,"OMP_NUM_THREADS"=>string(num_threads)),stdin=A,stderr=errmsg))
    catch
	@error String(readavailable(seekstart(errmsg)))
    end
    parse(Int,match(r"rank = ([0-9]*)",readlines(seekstart(errmsg))[end])[1])
end

################################################################
# one-stop shop for kernel and computation
kernel(A::CSR{F}; kwargs...) where F = kernel(echelonize(A; kwargs...))

LinearAlgebra.rank(A::CSR{F}; kwargs...) where F = rank(echelonize(A; kwargs...))

function pivots(qinv,k)
    pivots = Set(Int32(i-1) for i=1:size(k,2) if qinv[i]==-1)
    
    kj = unsafe_wrap(Vector{Int32},_get(k).j,nnz(k))

    spivpos = Int32[]
    for j=kj
        j∈pivots && push!(spivpos,j+1)
    end
    
    spivpos
end

function kernel_pivots(A::CSR{F}; kwargs...) where F
    e = echelonize(A; kwargs...)

    qi = qinv(e)
    k = kernel(e)
    (k, pivots(qinv(e),k))
end

################################################################
# block matrices and LU decompositions
include("blocks.jl")


# also, hook into gesv to avoid the triplet-to-matrix conversion twice

end
