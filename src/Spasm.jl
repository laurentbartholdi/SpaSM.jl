module Spasm

using SparseArrays

import Base: unsafe_convert
import SparseArrays: nnz

const spasm_lib = const LIB_FILE = "$(@__DIR__)" * "/../deps/spasm/src/.libs/libspasm.dylib"

struct GFp v::Cuint end
GFp(x::Integer,prime = 42013) = GFp(Cuint(mod(x,prime)))
GFp(x::GFp) = x
Base.convert(::Type{GFp},x) = GFp(x)
Base.convert(::Type{Int},x::GFp) = Int(x.v)
Base.zero(::GFp) = GFp(0)
Base.zero(::Type{GFp}) = GFp(0)
Base.one(::GFp) = GFp(1)
Base.one(::Type{GFp}) = GFp(1)
Base.show(io::IO,x::GFp) = print(io,"\e[1m",x.v,"\e[0m")
Cuint(x::GFp) = x.v
Int(x::GFp) = Int(x.v)

mutable struct spasm
    nzmax::Cint
    n::Cint # number of rows
    m::Cint # number of colums
    p::Ptr{Cint} # 0-based starts of rows
    j::Ptr{Cint} # 0-based column indices
    x::Ptr{GFp} # nonzeros
    prime::Cint
    function spasm(A::SparseMatrixCSC,prime::Int = 42013) # actually create transposed matrix
        nzmax = length(A.nzval)
        m,n = size(A)
        spasmA = csr_alloc(n,m,nzmax,prime,true)
        for i=1:length(A.colptr)
            unsafe_store!(spasmA.p,A.colptr[i]-1,i)
        end
        for i=1:length(A.rowval)
            unsafe_store!(spasmA.j,A.rowval[i]-1,i)
            unsafe_store!(spasmA.x,mod(A.nzval[i],prime),i)
        end
        finalizer(csr_free,spasmA)
        spasmA
    end
end

Base.show(io::IO, A::spasm) = print(io,A.n,"×",A.m," Spasm matrix % ",A.prime," with ",nnz(A)," (maximum ",A.nzmax,") non-zeros")

Base.size(A::spasm) = (A.n,A.m)

function SparseArrays.sparse(A::spasm)
    nnz = unsafe_load(A.p,A.n+1)
    SparseMatrixCSC{GFp,Int}(A.m,A.n,[Int(unsafe_load(A.p,i))+1 for i=1:A.n+1],[Int(unsafe_load(A.j,i))+1 for i=1:nnz],[unsafe_load(A.x,i) for i=1:nnz])
end

unsafe_load1(x) = unsafe_load(x,1)

# from spasm_util.c

nnz(A::spasm) = Int(@ccall spasm_lib."spasm_nnz"(pointer_from_objref(A)::Ptr{spasm})::Cint)

csr_alloc(m,n,nzmax,prime,with_values) = unsafe_load1(@ccall spasm_lib."spasm_csr_alloc"(m::Cint,n::Cint,nzmax::Cint,prime::Cint,with_values::Cint)::Ptr{spasm})

csr_realloc(A::spasm,nzmax) = @ccall spasm_lib."spasm_csr_realloc"(pointer_from_objref(A)::Ptr{spasm},nzmax::Cint)::Cvoid

csr_resize(A::spasm,m,n) = @ccall spasm_lib."spasm_csr_resize"(pointer_from_objref(A)::Ptr{spasm},m::Cint,n::Cint)::Cvoid

# cannot directly call csr_free because the spasm structure itself belongs to Julia
csr_free(A::spasm) = (@ccall free(A.p::Ptr{Cvoid})::Cvoid; @ccall free(A.j::Ptr{Cvoid})::Cvoid; @ccall free(A.x::Ptr{Cvoid})::Cvoid)

# declared, not defined
# identity(n,prime) = (A = unsafe_load1(@ccall spasm_lib."spasm_identity"(n::Cint,prime::Cint)::Ptr{spasm}); finalizer(csr_free,A); A)

get_num_threads() = Int(@ccall spasm_lib."spasm_get_num_threads"()::Cint)
get_thread_num() = Int(@ccall spasm_lib."spasm_get_thread_num"()::Cint)

# from spasm_transpose.c

Base.transpose(A::spasm) = unsafe_load1(@ccall spasm_lib."spasm_transpose"(pointer_from_objref(A)::Ptr{spasm})::Ptr{spasm})

# from spasm_submatrix.c

submatrix(A::spasm,r::UnitRange{Int},c::UnitRange{Int},with_values) = unsafe_load1(@ccall spasm_lib."spasm_submatrix"(pointer_from_objref(A)::Ptr{spasm},r.start::Cint,r.stop::Cint,c.start::Cint,c.stop::Cint,with_value::Cint)::Ptr{spasm})

sorted_submatrix(A::spasm,r::UnitRange{Int},c::UnitRange{Int},with_values) = unsafe_load1(@ccall spasm_lib."sorted_spasm_submatrix"(pointer_from_objref(A)::Ptr{spasm},r.start::Cint,r.stop::Cint,c.start::Cint,c.stop::Cint,with_value::Cint)::Ptr{spasm})

rows_submatrix(A::spasm,r::UnitRange{Int},with_values) = unsafe_load1(@ccall spasm_lib."spasm_rows_submatrix"(pointer_from_objref(A)::Ptr{spasm},r.start::Cint,r.stop::Cint,with_value::Cint)::Ptr{spasm})

"""
# from spasm_permutation.c
void spasm_pvec(const int *p, const spasm_GFp * b, spasm_GFp * x, int n);
void spasm_ipvec(const int *p, const spasm_GFp * b, spasm_GFp * x, int n);
int *spasm_pinv(int const *p, int n);
spasm *spasm_permute(const spasm * A, const int *p, const int *qinv, int with_values);
int *spasm_random_permutation(int n);
void spasm_range_pvec(int *x, int a, int b, int *p);
"""

# from spasm_GFp.c
inverse(a::GFp,prime) = @ccall spasm_lib."spasm_GFp_inverse"(a::GFp,prime::Cint)::GFp

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

/* spasm_triangular.c */
int spasm_is_upper_triangular(const spasm * A);
int spasm_is_lower_triangular(const spasm * A);
void spasm_dense_back_solve(const spasm * L, spasm_GFp * b, spasm_GFp * x, const int *p);
int spasm_dense_forward_solve(const spasm * U, spasm_GFp * b, spasm_GFp * x, const int *q);
int spasm_sparse_backward_solve(const spasm * L, const spasm * B, int k, int *xi, spasm_GFp * x, const int *pinv, int r_b
ound);
int spasm_sparse_forward_solve(const spasm * U, const spasm * B, int k, int *xi, spasm_GFp * x, const int *pinv);

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

/* spasm_cc.c */
spasm_dm *spasm_connected_components(const spasm * A, spasm * given_At);
"""

# from spasm_kernel.c
kernel(A,column_permutation = nothing) = unsafe_load1(@ccall spasm_lib."spasm_kernel"(pointer_from_objref(A)::Ptr{spasm},(column_permutation == nothing ? C_NULL : pointer(column_permutation))::Ptr{Cint})::Ptr{spasm})

"""
/* spasm_uetree.c */
int * spasm_uetree(const spasm * A);
int *spasm_tree_postorder(const spasm *A, const int *parent);
int *spasm_tree_topological_postorder(const spasm *A, const int *parent);

"""

end
