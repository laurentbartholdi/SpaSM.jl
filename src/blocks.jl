struct Block{T}
    blocks::Vector{T} # the blocks themselves
    row2block::Vector{Tuple{Int32,Int32}} # column to (block number, pos in block). pos is 0-based
    col2block::Vector{Tuple{Int32,Int32}} # column to (block number, pos in block)
    block2row::Vector{Vector{Int32}} # block2col[i][j] = jth element of ith block. also 0-based
    block2col::Vector{Vector{Int32}} # block2col[i][j] = jth element of ith block
end

Base.length(block::Block) = length(block.blocks)

Base.size(block::Block) = (length(block.row2block),length(block.col2block))
Base.size(block::Block, i::Int) = size(block)[i]

function Base.show(io::IO, block::Block{T}) where T
    for b=1:length(block)
        print(io,"$(block.block2row[b])×$(block.block2col[b]): $(block.blocks[b])\n")
    end
end

function Base.getproperty(block::Block{LU{F}},s::Symbol) where F
    if s===:U
        return Block([X.U for X=block.blocks],block.row2block,block.col2block,block.block2row,block.block2col)
    elseif s===:L
        return Block([X.L for X=block.blocks],block.row2block,block.col2block,block.block2row,block.block2col)
    else
        return getfield(block,s)
    end
end

"""
    Block(A::CSR{F}) where F

Convert the matrix `A` to block form, by studying its nonzero pattern
"""
function Block(A::CSR{F}) where F
    rowptr = A.p
    colval = A.j
    nzval = A.x
    m, n = Int32.(size(A))

    rowcol = IntDisjointSets{Int32}(m+n)

    for i=Int32(1):m
        for j=rowptr[i]+1:rowptr[i+1]
            union!(rowcol,i,colval[j]+m+Int32(1))
        end
    end
    roots, num_roots = let num2block = zeros(Int32,m+n)
        num_roots = 0
        for i=Int32(1):m+n
            if i == find_root!(rowcol,i)
                num_roots += 1
                num2block[i] = num_roots
            end
        end
        for i=Int32(1):m+n
            num2block[i] = num2block[find_root!(rowcol,i)]
        end
        num2block, num_roots
    end
    
    col2block = Tuple{Int32,Int32}[]
    row2block = Tuple{Int32,Int32}[]
    block2col = [Int32[] for _=1:num_roots]
    block2row = [Int32[] for _=1:num_roots]

    for i=Int32(1):m
        blocknum = roots[i]
        push!(row2block,(blocknum,length(block2row[blocknum])))
        push!(block2row[blocknum],i-1)
    end
    for j=Int32(1):n
        blocknum = roots[j+m]
        push!(col2block,(blocknum,length(block2col[blocknum])))
        push!(block2col[blocknum],j-1)
    end

    blocks = CSR{F}[]

    block_nnz = zeros(Int,num_roots)
    for i=1:m
        block_nnz[row2block[i][1]] += rowptr[i+1]-rowptr[i]
    end
    for b=1:num_roots
        nnz = 0
        for i=block2row[b]
            nnz += rowptr[i+2]-rowptr[i+1]
        end
        mat = csr_alloc(length(block2row[b]),length(block2col[b]),nnz,F.p)
        subrowptr = mat.p
        subcolval = mat.j
        subnzval = mat.x
        subnnz = 0
        for (subi,i)=enumerate(block2row[b])
            for j=rowptr[i+1]+1:rowptr[i+2]
                subnnz += 1
                subcolval[subnnz] = col2block[colval[j]+1][2]
                subnzval[subnnz] = nzval[j]
            end
            subrowptr[subi+1] = subnnz
        end
        push!(blocks,mat)
    end
    Block{CSR{F}}(blocks,row2block,col2block,block2row,block2col)
end

function echelonize(block::Block{CSR{F}}; kwargs...) where F
    lu = Vector{LU{F}}(undef,length(block))
    
    for b=1:length(block)
        lu[b] = echelonize(block.blocks[b]; kwargs...)
    end

    Block{LU{F}}(lu,block.row2block,block.col2block,block.block2row,block.block2col)
end

LinearAlgebra.rank(block::Block; kwargs...) = sum(rank(X; kwargs...) for X=block.blocks;init=0)
    
function kernel(block::Block{LU{F}}; kwargs...) where F
    k = Vector{CSR{F}}(undef,length(block))
    
    for b=1:length(block)
        k[b] = kernel(block.blocks[b]; kwargs...)
    end
    block2row = Vector{Int32}[]
    row2block = Tuple{Int32,Int32}[]
    rank = 0
    for b=1:length(block)
        subrank = size(k[b],1)
        push!(block2row,rank:rank+subrank-1)
        for i=1:subrank
            push!(row2block,(b,i-1))
        end
        rank += subrank
    end
    Block{CSR{F}}(k,row2block,block.col2block,block2row,block.block2col)
end

kernel(block::Block{CSR{F}}; kwargs...) where F = kernel(echelonize(block; kwargs...))

transpose(block::Block{CSR{F}}) where F = Block{LU{F}}([transpose(X) for X=block.blocks],block.col2block,block.row2block,block.block2col,block.block2row)

function CSR(block::Block{CSR{F}}) where F
    nnz = 0
    for b=1:length(block)
        nnz += Spasm.nnz(block.blocks[b])
    end

    m, n = size(block)

    A = csr_alloc(m,n,nnz,F.p)
    rowptr = A.p
    colval = A.j
    nzval = A.x

    nnz = 0
    for i=1:m
        b, subi = block.row2block[i]
        subrowptr = block.blocks[b].p
        subcolval = block.blocks[b].j
        subnzval = block.blocks[b].x
        for j=subrowptr[subi+1]+1:subrowptr[subi+2]
            nnz += 1
            colval[nnz] = block.block2col[b][subcolval[j]+1]
            nzval[nnz] = subnzval[j]
        end
        rowptr[i+1] = nnz
    end
    A
end

"""
    sparse_triangular_solve(block::Block{LU{F}}, m::CSR{F})

Simultaneously solve the block system `X`*`block.blocks` == `B`.
`Returns 
"""
function sparse_triangular_solve(block::Block{LU{F}}, B::CSR{F}) where F
    @assert size(block,2)==size(B,2)

    # slightly ugly: we construct a SparseMatrixCSC because it's more
    # Julia-friendly, we convert it at the end to a CSR matrix
    
    Xt = spzeros(ZZp{F},size(block,1),size(B,1)) # transposed solution

    rowptr = B.p
    colval = B.j
    nzval = B.x

    ms = [size(block.blocks[b].U,2) for b=1:length(block)]
    xjs = [Vector{Int32}(undef,3m) for m=ms]
    xs = [Vector{ZZp{F}}(undef,m) for m=ms]
    Bs = [csr_alloc(1,m,m,F.p) for m=ms]
    rowptrs = [B.p for B=Bs]
    colvals = [B.j for B=Bs]
    nzvals = [B.x for B=Bs]
    for k=1:size(B,1)
        for b=1:length(block)
            rowptrs[b][2] = 0 # empty matrices Bs
        end
        for p=rowptr[k]+1:rowptr[k+1]
            # split B[k,:] into matrices Bs
            j = colval[p]
            (b,subj) = block.col2block[j+1]
            rowptrs[b][2] += 1
            colvals[b][rowptrs[b][2]] = subj
            nzvals[b][rowptrs[b][2]] = nzval[p]
        end
        for b=1:length(block)
            xj = xjs[b]
            x = xs[b]
            qinv = block.blocks[b].qinv
            fill!(xj,0)
            top = sparse_triangular_solve(block.blocks[b].U, Bs[b], 0, xj, x, qinv)
            for i=top+1:ms[b]
                j = xj[i]+1
                iszero(x[j]) && continue
                qinv[j]<0 && return nothing
                push!(Xt.rowval,block.block2row[b][qinv[j]+1]+1)
                push!(Xt.nzval,x[j])
            end
        end
        Xt.colptr[k+1] = length(Xt.rowval)+1
    end
    CSR(Xt)
end
