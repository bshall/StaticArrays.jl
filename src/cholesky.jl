# Generic Cholesky decomposition for fixed-size matrices, mostly unrolled
@inline function Base.chol(A::StaticMatrix)
    ishermitian(A) || Base.LinAlg.non_hermitian_error("chol")
    _chol(Size(A), A, UpperTriangular)
end

@inline function Base.chol(A::Base.LinAlg.RealHermSymComplexHerm{<:Real, <:StaticMatrix})
    _chol(Size(A), A.uplo == 'U' ? A.data : ctranspose(A.data), UpperTriangular)
end

@inline function Base.cholfact(A::StaticMatrix, uplo::Symbol, ::Type{Val{false}})
    ishermitian(A) || Base.LinAlg.non_hermitian_error("cholfact")
    Base.cholfact(Hermitian(A, uplo), Val{false})
end

@inline function Base.cholfact(A::StaticMatrix, uplo::Symbol = :U)
    ishermitian(A) || Base.LinAlg.non_hermitian_error("cholfact")
    Base.cholfact(Hermitian(A, uplo))
end

@inline function Base.cholfact(A::Base.LinAlg.RealHermSymComplexHerm{<:Real, <:StaticMatrix}, ::Type{Val{false}})
    if A.uplo == 'U'
        CU = _chol(Size(A), A.data, UpperTriangular)
        Base.LinAlg.Cholesky(CU.data, 'U')
    else
        CL = _chol(Size(A), A.data, LowerTriangular)
        Base.LinAlg.Cholesky(CL.data, 'L')
    end
end

@inline function Base.cholfact(A::Base.LinAlg.RealHermSymComplexHerm{<:Real, <:StaticMatrix}) = cholfact(A, Val{false})

@generated function _chol(::Size{s}, a::StaticMatrix{<:Any, <:Any, T}, ::Type{UpperTriangular}) where {s, T}
    if s[1] != s[2]
        error("matrix must be square")
    end
    n = s[1]
    Tx = promote_type(typeof(sqrt(one(T))), Float32)

    x = [i > j ? Symbol("x_0") : Symbol("x_$(i)_$(j)") for i = 1:n, j = 1:n]

    code = quote x_0 = $(zero(Tx)) end
    for k = 1:n
        ex = :(a[$k, $k])
        for i = 1:k-1
            ex = :($ex - $(x[i, k])'*$(x[i, k]))
        end
        push!(code.args, :($(x[k, k]) = sqrt($ex)))
        if k < n
            push!(code.args, :(xkkInv = inv($(x[k, k])')))
        end
        for j = k + 1:n
            ex = :(a[$k,$j])
            for i = 1:k-1
                ex = :($ex - $(x[i, k])'*$(x[i, j]))
            end
            push!(code.args, :($(x[k, j]) = $ex*xkkInv))
        end
    end

    quote
        @_inline_meta
        @inbounds $code
        @inbounds return UpperTriangular(similar_type(a, $Tx)(tuple($(x...))))
    end
end

@generated function _chol(::Size{s}, a::StaticMatrix{<:Any, <:Any, T}, ::Type{LowerTriangular}) where {s, T}
    if s[1] != s[2]
        error("matrix must be square")
    end

    n = s[1]
    Tx = promote_type(typeof(sqrt(one(T))), Float32)

    x = [i < j ? Symbol("x_0") : Symbol("x_$(i)_$(j)") for i = 1:n, j = 1:n]

    code = quote x_0 = $(zero(Tx)) end
    for k = 1:n
        ex = :(a[$k, $k])
        for i = 1:k-1
            ex = :($ex - $(x[k, i])*$(x[k, i])')
        end
        push!(code.args, :($(x[k, k]) = sqrt($ex)))
        push!(code.args, :(xkkInv = inv($(x[k, k]'))))
        for i = k+1:n
            ex = :(a[$i, $k])
            for j = 1:k-1
                ex = :($ex - $(x[i, j])*$(x[k, j])'*xkkInv')
            end
            ex = :($ex*xkkInv')
            push!(code.args, :($(x[i, k]) = $ex*xkkInv'))
        end
    end

    quote
        @_inline_meta
        @inbounds $code
        @inbounds return LowerTriangular(similar_type(a, $Tx)(tuple($(x...))))
    end
end
