# Generic Cholesky decomposition for fixed-size matrices, mostly unrolled
@inline function Base.chol(A::StaticMatrix)
    ishermitian(A) || Base.LinAlg.non_hermitian_error("chol")
    C, info = _chol(Size(A), A, UpperTriangular)
    Base.LinAlg.@assertposdef C info
end

@inline function Base.chol(A::Base.LinAlg.RealHermSymComplexHerm{<:Real,<:StaticMatrix})
    C, info = _chol(Size(A), A.uplo == 'U' ? A.data : ctranspose(A.data), UpperTriangular)
    Base.LinAlg.@assertposdef C info
end

@inline function Base.cholfact(A::Base.LinAlg.RealHermSymComplexHerm{<:Real,<:StaticMatrix}, ::Type{Val{false}}=Val{false})
    if A.uplo == 'U'
        CU, info = _chol(Size(A), A.data, UpperTriangular)
        # TODO include info in v0.7
        Base.LinAlg.Cholesky(CU.data, 'U')
    else
        CL, info = _chol(Size(A), A.data, LowerTriangular)
        # TODO include info in v0.7
        Base.LinAlg.Cholesky(CL.data, 'L')
    end
end

@inline function Base.cholfact(A::StaticMatrix, ::Type{Val{false}}=Val{false})
    ishermitian(A) || Base.LinAlg.non_hermitian_error("cholfact")
    Base.cholfact(Hermitian(A), Val{false})
end

@generated function _chol(::Size{s}, A::StaticMatrix{<:Any,<:Any,T}, ::Type{UpperTriangular}) where {s, T}
    if s[1] != s[2]
        error("matrix must be square")
    end
    n = s[1]
    TX = promote_type(typeof(chol(one(T), UpperTriangular)), Float32)

    X = [Symbol("X_$(i)_$(j)") for i = 1:n, j = 1:n]
    init = [:($(X[i,j]) = A[$i,$j]) for i = 1:n, j = 1:n]

    code = quote end
    for k = 1:n
        ex = :($(X[k,k]))
        for i = 1:k-1
            ex = :($ex - $(X[i,k])'*$(X[i,k]))
        end
        push!(code.args, quote $(X[k,k]), info = _chol($ex, UpperTriangular) end)
        push!(code.args, :(info == 0 || return UpperTriangular(similar_type(A, $TX)(tuple($(X...)))), info))
        if k < n
            push!(code.args, :(XkkInv = inv($(X[k,k])')))
        end
        for j = k + 1:n
            ex = :($(X[k,j]))
            for i = 1:k-1
                ex = :($ex - $(X[i,k])'*$(X[i,j]))
            end
            push!(code.args, :($(X[k,j]) = XkkInv*$ex))
        end
    end

    quote
        @_inline_meta
        @inbounds $(Expr(:block, init...))
        @inbounds $code
        @inbounds return UpperTriangular(similar_type(A, $TX)(tuple($(X...)))), convert(Base.LinAlg.BlasInt, 0)
    end
end

@generated function _chol(::Size{s}, A::StaticMatrix{<:Any, <:Any, T}, ::Type{LowerTriangular}) where {s, T}
    if s[1] != s[2]
        error("matrix must be square")
    end

    n = s[1]
    TX = promote_type(typeof(chol(one(T), LowerTriangular)), Float32)

    X = [Symbol("X_$(i)_$(j)") for i = 1:n, j = 1:n]
    init = [:($(X[i,j]) = A[$i,$j]) for i = 1:n, j = 1:n]

    code = quote end
    for k = 1:n
        ex = :($(X[k,k]))
        for i = 1:k-1
            ex = :($ex - $(X[k,i])*$(X[k,i])')
        end
        push!(code.args, quote $(X[k,k]), info = _chol($ex, LowerTriangular) end)
        push!(code.args, :(info == 0 || return LowerTriangular(similar_type(A, $TX)(tuple($(X...)))), info))
        if k < n
            push!(code.args, :(XkkInv = inv($(X[k,k])')))
        end
        for j = 1:k-1
            for i = k+1:n
                push!(code.args, :($(X[i,k]) -= $(X[i,j])*$(X[k,j])'))
            end
        end
        for i = k+1:n
            push!(code.args, :($(X[i,k]) *= XkkInv))
        end
    end

    quote
        @_inline_meta
        @inbounds $(Expr(:block, init...))
        @inbounds $code
        @inbounds return LowerTriangular(similar_type(A, $TX)(tuple($(X...)))), convert(Base.LinAlg.BlasInt, 0)
    end
end

## Numbers
function _chol(x::Number, uplo)
    rx = real(x)
    rxr = sqrt(abs(rx))
    rval = convert(promote_type(typeof(x), typeof(rxr)), rxr)
    rx == abs(x) ? (rval, convert(Base.LinAlg.BlasInt, 0)) : (rval, convert(Base.LinAlg.BlasInt, 1))
end

chol(x::Number, uplo) = ((C, info) = _chol(x, uplo); Base.LinAlg.@assertposdef C info)

function Base.:\(C::Base.LinAlg.Cholesky{<:Any,<:StaticMatrix}, B::StaticVecOrMat)
    if C.uplo == 'L'
        return LowerTriangular(C.factors)' \ (LowerTriangular(C.factors) \ B)
    else
        return UpperTriangular(C.factors) \ (UpperTriangular(C.factors)' \ B)
    end
end

@inline Base.det(C::Base.LinAlg.Cholesky{<:Any,<:StaticMatrix}) = _det(Size(C.factors), C)

@generated function _det(::Size{s}, C::Base.LinAlg.Cholesky{T,<:StaticMatrix}) where {s,T}
    # TODO add following line for v0.7
    # C.info == 0 || throw(PosDefException(C.info))

    ex = :(one(real($T)))
    for i = 1:s[1]
        ex = :($ex*real(C.factors[$i,$i])^2)
    end

    @inbounds return ex
end
