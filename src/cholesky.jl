# Generic Cholesky decomposition for fixed-size matrices, mostly unrolled
@inline function Base.chol(A::StaticMatrix)
    ishermitian(A) || Base.LinAlg.non_hermitian_error("chol")
    _chol(Size(A), A)
end

@inline function Base.chol(A::Base.LinAlg.RealHermSymComplexHerm{<:Real, <:StaticMatrix})
    _chol(Size(A), A.data)
end

@generated function _chol(::Size{s}, a::StaticMatrix{<:Any, <:Any, T}) where {s, T}
    if s[1] != s[2]
        error("matrix must be square")
    end
    n = s[1]
    Tx = promote_type(typeof(sqrt(one(T))), Float32)

    x = [i > j ? Symbol("x_0") : Symbol("x_$(i)_$(j)") for i = 1:n, j = 1:n]

    code = quote x_0 = $(zero(Tx)) end
    for k = 1:n
        ex = :(a[$k,$k])
        for i = 1:k-1
            ex = :($ex - $(x[i,k])'*$(x[i,k]))
        end
        push!(code.args, :($(x[k,k]) = sqrt($ex)))
        if k < n
            push!(code.args, :(xkkInv = inv($(x[k,k])')))
        end
        for j = k + 1:n
            ex = :(a[$k,$j])
            for i = 1:k-1
                ex = :($ex - $(x[i,k])'*$(x[i,j]))
            end
            ex = :($ex*xkkInv)
            push!(code.args, :($(x[k,j]) = $ex))
        end
    end

    quote
        $(Expr(:meta, :inline))
        @inbounds $code
        @inbounds return similar_type(a, $Tx)(tuple($(x...)))
    end
end
