using Base.LinAlg: BlasInt

struct LU{T,S<:StaticMatrix} <: Factorization{T}
    factors::S
    ipiv::SVector{BlasInt}
    info::BlasInt
    LU{T,S}(factors::StaticMatrix{T}, ipiv::SVector{BlasInt}, info::BlasInt) where {T,S} = new(factors, ipiv, info)
end
LU(factors::StaticMatrix{T}, ipiv::SVector{BlasInt}, info::BlasInt) where {T} = LU{T,typeof(factors)}(factors, ipiv, info)

@inline lufact(A::StaticMatrix, pivot::Union{Val{false}, Val{true}} = Val(true)) = _lufact(Size(A), A, pivot)

@generated function _lufact(::Size{s}, a::StaticMatrix{<:Any,<:Any,T}, ::Val{Pivot}) where {s,T,Pivot}
    m, n = s
    minmn = min(m,n)
    X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]
    ipiv = [Symbol("p_$i") for i = 1:minmn]

    TX = typeof(zero(T)/one(T))

    init = [:($(X[i,j]) = A[$(sub2ind(s,i,j))]) for i = 1:m, j = 1:n]
    code = quote info = 0 end
    for k = 1:minmn
        if Pivot
            ex = :(indmax(tuple($([:(abs($(X[i,k]))) for i = k:m]...))))
            push!(code.args, :($(ipiv[k]) = $k - 1 + $ex))
        else
            push!(code.args, :($(ipiv[k]) = $k))
        end
        for kp = k:m
            swap = quote end
            if kp > k
                for i = 1:n
                    push!(swap.args, :(tmp = $(X[k,i])))
                    push!(swap.args, :($(X[k,i]) = $(X[kp,i])))
                    push!(swap.args, :($(X[kp,i]) = tmp))
                end
            end
            push!(swap.args, :(xkkInv = inv($(X[k,k]))))
            for i = k+1:m
                push!(swap.args, :($(X[i,k]) *= xkkInv))
            end
            ex = quote
                if $(ipiv[k]) == $kp
                    if !iszero($(X[kp,k]))
                        $swap
                    elseif info == 0
                        info = $k
                    end
                end
            end
            push!(code.args, ex)
        end
        update = quote end
        for j = k+1:n
            for i = k+1:m
                push!(update.args, :($(X[i,j]) -= $(X[i,k])*$(X[k,j])))
            end
        end
        push!(code.args, update)
    end

    quote
        $(Expr(:meta, :inline))
        @inbounds $(Expr(:block, init...))
        @inbounds $code
        @inbounds return LU(similar_type(a, $TX)(tuple($(X...))), SVector{$minmn,BlasInt}(tuple($(ipiv...))), convert(BlasInt, info))
    end
end

# LU decomposition
function lu(A::StaticMatrix, pivot::Union{Type{Val{false}},Type{Val{true}}}=Val{true})
    L,U,p = _lu(Size(A), A, pivot)
    (L,U,p)
end

# For the square version, return explicit lower and upper triangular matrices.
# We would do this for the rectangular case too, but Base doesn't support that.
function lu(A::StaticMatrix{N,N}, pivot::Union{Type{Val{false}},Type{Val{true}}}=Val{true}) where {N}
    L,U,p = _lu(Size(A), A, pivot)
    (LowerTriangular(L), UpperTriangular(U), p)
end


@inline function _lu(::Size{S}, A::StaticMatrix, pivot) where {S}
    # For now, just call through to Base.
    # TODO: statically sized LU without allocations!
    f = lufact(Matrix(A), pivot)
    T = eltype(A)
    # Trick to get the output eltype - can't rely on the result of f[:L] as
    # it's not type inferrable.
    T2 = typeof((one(T)*zero(T) + zero(T))/one(T))
    L = similar_type(A, T2, Size(Size(A)[1], diagsize(A)))(f[:L])
    U = similar_type(A, T2, Size(diagsize(A), Size(A)[2]))(f[:U])
    p = similar_type(A, Int, Size(Size(A)[1]))(f[:p])
    (L,U,p)
end

# Base.lufact() interface is fairly inherently type unstable.  Punt on
# implementing that, for now...
