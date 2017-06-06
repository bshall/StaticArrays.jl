import Base: *, Ac_mul_B, At_mul_B, A_mul_Bc, A_mul_Bt, At_mul_Bt, Ac_mul_Bc
import Base: triu, tril

@inline Size(A::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}) = Size(A.data)

@inline *(A::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}, B::StaticVecOrMat) = _A_mul_B(Size(A), Size(B), A, B)
@inline *(A::StaticMatrix, B::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}) = _A_mul_B(Size(A), Size(B), A, B)
@inline Ac_mul_B(A::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}, B::StaticVecOrMat) = _Ac_mul_B(Size(A), Size(B), A, B)
@inline A_mul_Bc(A::StaticMatrix, B::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}) = _A_mul_Bc(Size(A), Size(B), A, B)
@inline At_mul_B(A::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}, B::StaticVecOrMat) = _At_mul_B(Size(A), Size(B), A, B)
@inline A_mul_Bt(A::StaticMatrix, B::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}) = _A_mul_Bt(Size(A), Size(B), A, B)

# Specializations for RowVector
@inline *(rowvec::RowVector{<:Any,<:StaticVector}, A::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}) = transpose(A * transpose(rowvec))
@inline A_mul_Bt(rowvec::RowVector{<:Any,<:StaticVector}, A::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}) = transpose(A * transpose(rowvec))
@inline A_mul_Bt(A::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}, rowvec::RowVector{<:Any,<:StaticVector}) = A * transpose(rowvec)
@inline At_mul_Bt(A::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}, rowvec::RowVector{<:Any,<:StaticVector}) = A.' * transpose(rowvec)
@inline A_mul_Bc(rowvec::RowVector{<:Any,<:StaticVector}, A::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}) = ctranspose(A * ctranspose(rowvec))
@inline A_mul_Bc(A::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}, rowvec::RowVector{<:Any,<:StaticVector}) = A * ctranspose(rowvec)
@inline Ac_mul_Bc(A::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}, rowvec::RowVector{<:Any,<:StaticVector}) = A' * ctranspose(rowvec)

Ac_mul_B(A::StaticMatrix, B::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}) = (*)(ctranspose(A), B)
At_mul_B(A::StaticMatrix, B::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}) = (*)(transpose(A), B)
A_mul_Bc(A::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}, B::StaticMatrix) = (*)(A, ctranspose(B))
A_mul_Bt(A::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}, B::StaticMatrix) = (*)(A, transpose(B))
# Ac_mul_Bc(A::AbstractTriangular, B::AbstractTriangular) = Ac_mul_B(A, B')
Ac_mul_Bc(A::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}, B::StaticMatrix) = Ac_mul_B(A, B')
Ac_mul_Bc(A::StaticMatrix, B::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}) = A_mul_Bc(A', B)
# At_mul_Bt(A::AbstractTriangular, B::AbstractTriangular) = At_mul_B(A, B.')
At_mul_Bt(A::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}, B::StaticMatrix) = At_mul_B(A, B.')
At_mul_Bt(A::StaticMatrix, B::Base.LinAlg.AbstractTriangular{<:Any,<:StaticMatrix}) = A_mul_Bt(A.', B)

@generated function _A_mul_B(::Size{sa}, ::Size{sb}, A::UpperTriangular{Ta,<:StaticMatrix}, B::StaticVecOrMat{Tb}) where {sa,sb,Ta,Tb}
    # print("1")
    m = sb[1]
    n = length(sb) > 1 ? sb[2] : 1
    if m != sa[1]
        throw(DimensionMismatch("right hand side B needs first dimension of size $(sa[1]), has size $m"))
    end

    T = promote_matprod(Ta, Tb)
    X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

    code = quote end
    for j = 1:n
        for i = 1:m
            ex = :(A.data[$i,$i]*B[$i,$j])
            for k = i + 1:m
                ex = :($ex + A.data[$i,$k]*B[$k,$j])
            end
            push!(code.args, :($(X[i,j]) = $ex))
        end
    end

    return quote
        @_inline_meta
        @inbounds $code
        return similar_type(B, $T)(tuple($(X...)))
    end
end

@generated function _A_mul_B(::Size{sa}, ::Size{sb}, A::LowerTriangular{Ta, <:StaticMatrix}, B::StaticVecOrMat{Tb}) where {sa,sb,Ta,Tb}
    # print("2")
    m = sb[1]
    n = length(sb) > 1 ? sb[2] : 1
    if m != sa[1]
        throw(DimensionMismatch("right hand side B needs first dimension of size $(sa[1]), has size $m"))
    end

    T = promote_matprod(Ta, Tb)
    X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

    code = quote end
    for j = 1:n
        for i = m:-1:1
            ex = :(A.data[$i,$i]*B[$i,$j])
            for k = 1:i - 1
                ex = :($ex + A.data[$i,$k]*B[$k,$j])
            end
            push!(code.args, :($(X[i,j]) = $ex))
        end
    end

    return quote
        @_inline_meta
        @inbounds $code
        return similar_type(B, $T)(tuple($(X...)))
    end
end

@generated function _Ac_mul_B(::Size{sa}, ::Size{sb}, A::UpperTriangular{Ta, <:StaticMatrix}, B::StaticVecOrMat{Tb}) where {sa,sb,Ta,Tb}
    # print("3")
    m = sb[1]
    n = length(sb) > 1 ? sb[2] : 1
    if m != sa[1]
        throw(DimensionMismatch("right hand side B needs first dimension of size $(sa[1]), has size $m"))
    end

    T = promote_matprod(Ta, Tb)
    X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

    code = quote end
    for j = 1:n
        for i = m:-1:1
            ex = :(A.data[$i,$i]'*B[$i,$j])
            for k = 1:i - 1
                ex = :($ex + A.data[$k,$i]'*B[$k,$j])
            end
            push!(code.args, :($(X[i,j]) = $ex))
        end
    end

    return quote
        @_inline_meta
        @inbounds $code
        return similar_type(B, $T)(tuple($(X...)))
    end
end

@generated function _Ac_mul_B(::Size{sa}, ::Size{sb}, A::LowerTriangular{Ta, <:StaticMatrix}, B::StaticVecOrMat{Tb}) where {sa,sb,Ta,Tb}
    # print("4")
    m = sb[1]
    n = length(sb) > 1 ? sb[2] : 1
    if m != sa[1]
        throw(DimensionMismatch("right hand side B needs first dimension of size $(sa[1]), has size $m"))
    end

    T = promote_matprod(Ta, Tb)
    X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

    code = quote end
    for j = 1:n
        for i = 1:m
            ex = :(A.data[$i,$i]'*B[$i,$j])
            for k = i + 1:m
                ex = :($ex + A.data[$k,$i]'*B[$k,$j])
            end
            push!(code.args, :($(X[i,j]) = $ex))
        end
    end

    return quote
        @_inline_meta
        @inbounds $code
        return similar_type(B, $T)(tuple($(X...)))
    end
end

@generated function _At_mul_B(::Size{sa}, ::Size{sb}, A::UpperTriangular{Ta, <:StaticMatrix}, B::StaticVecOrMat{Tb}) where {sa,sb,Ta,Tb}
    # print("5")
    m = sb[1]
    n = length(sb) > 1 ? sb[2] : 1
    if m != sa[1]
        throw(DimensionMismatch("right hand side B needs first dimension of size $(sa[1]), has size $m"))
    end

    T = promote_matprod(Ta, Tb)
    X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

    code = quote end
    for j = 1:n
        for i = m:-1:1
            ex = :(A.data[$i,$i].'*B[$i,$j])
            for k = 1:i - 1
                ex = :($ex + A.data[$k,$i].'*B[$k,$j])
            end
            push!(code.args, :($(X[i,j]) = $ex))
        end
    end

    return quote
        @_inline_meta
        @inbounds $code
        return similar_type(B, $T)(tuple($(X...)))
    end
end

@generated function _At_mul_B(::Size{sa}, ::Size{sb}, A::LowerTriangular{Ta, <:StaticMatrix}, B::StaticVecOrMat{Tb}) where {sa,sb,Ta,Tb}
    # print("6")
    m = sb[1]
    n = length(sb) > 1 ? sb[2] : 1
    if m != sa[1]
        throw(DimensionMismatch("right hand side B needs first dimension of size $(sa[1]), has size $m"))
    end

    T = promote_matprod(Ta, Tb)
    X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

    code = quote end
    for j = 1:n
        for i = 1:m
            ex = :(A.data[$i,$i].'*B[$i,$j])
            for k = i + 1:m
                ex = :($ex + A.data[$k,$i].'*B[$k,$j])
            end
            push!(code.args, :($(X[i,j]) = $ex))
        end
    end

    return quote
        @_inline_meta
        @inbounds $code
        return similar_type(B, $T)(tuple($(X...)))
    end
end

@generated function _A_mul_B(::Size{sa}, ::Size{sb}, A::StaticMatrix{<:Any,<:Any,Ta}, B::UpperTriangular{Tb,<:StaticMatrix}) where {sa,sb,Ta,Tb}
    m, n = sa[1], sa[2]
    if sb[1] != n
        throw(DimensionMismatch("right hand side B needs first dimension of size $n, has size $(sb[1])"))
    end

    T = promote_matprod(Ta, Tb)
    X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

    code = quote end
    for i = 1:m
        for j = n:-1:1
            ex = :(A[$i,$j]*B[$j,$j])
            for k = 1:j - 1
                ex = :($ex + A[$i,$k]*B.data[$k,$j])
            end
            push!(code.args, :($(X[i,j]) = $ex))
        end
    end

    return quote
        @_inline_meta
        @inbounds $code
        return similar_type(A, $T)(tuple($(X...)))
    end
end

@generated function _A_mul_B(::Size{sa}, ::Size{sb}, A::StaticMatrix{<:Any,<:Any,Ta}, B::LowerTriangular{Tb,<:StaticMatrix}) where {sa,sb,Ta,Tb}
    m, n = sa[1], sa[2]
    if sb[1] != n
        throw(DimensionMismatch("right hand side B needs first dimension of size $n, has size $(sb[1])"))
    end

    T = promote_matprod(Ta, Tb)
    X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

    code = quote end
    for i = 1:m
        for j = 1:n
            ex = :(A[$i,$j]*B[$j,$j])
            for k = j + 1:n
                ex = :($ex + A[$i,$k]*B.data[$k,$j])
            end
            push!(code.args, :($(X[i,j]) = $ex))
        end
    end

    return quote
        @_inline_meta
        @inbounds $code
        return similar_type(A, $T)(tuple($(X...)))
    end
end

@generated function _A_mul_Bc(::Size{sa}, ::Size{sb}, A::StaticMatrix{<:Any,<:Any,Ta}, B::UpperTriangular{Tb,<:StaticMatrix}) where {sa,sb,Ta,Tb}
    m, n = sa[1], sa[2]
    if sb[1] != n
        throw(DimensionMismatch("right hand side B needs first dimension of size $n, has size $(sb[1])"))
    end

    T = promote_matprod(Ta, Tb)
    X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

    code = quote end
    for i = 1:m
        for j = 1:n
            ex = :(A[$i,$j]*B[$j,$j]')
            for k = j + 1:n
                ex = :($ex + A[$i,$k]*B.data[$j,$k]')
            end
            push!(code.args, :($(X[i,j]) = $ex))
        end
    end

    return quote
        @_inline_meta
        @inbounds $code
        return similar_type(A, $T)(tuple($(X...)))
    end
end

@generated function _A_mul_Bc(::Size{sa}, ::Size{sb}, A::StaticMatrix{<:Any,<:Any,Ta}, B::LowerTriangular{Tb,<:StaticMatrix}) where {sa,sb,Ta,Tb}
    m, n = sa[1], sa[2]
    if sb[1] != n
        throw(DimensionMismatch("right hand side B needs first dimension of size $n, has size $(sb[1])"))
    end

    T = promote_matprod(Ta, Tb)
    X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

    code = quote end
    for i = 1:m
        for j = n:-1:1
            ex = :(A[$i,$j]*B[$j,$j]')
            for k = 1:j - 1
                ex = :($ex + A[$i,$k]*B.data[$j,$k]')
            end
            push!(code.args, :($(X[i,j]) = $ex))
        end
    end

    return quote
        @_inline_meta
        @inbounds $code
        return similar_type(A, $T)(tuple($(X...)))
    end
end

@generated function _A_mul_Bt(::Size{sa}, ::Size{sb}, A::StaticMatrix{<:Any,<:Any,Ta}, B::UpperTriangular{Tb,<:StaticMatrix}) where {sa,sb,Ta,Tb}
    m, n = sa[1], sa[2]
    if sb[1] != n
        throw(DimensionMismatch("right hand side B needs first dimension of size $n, has size $(sb[1])"))
    end

    T = promote_matprod(Ta, Tb)
    X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

    code = quote end
    for i = 1:m
        for j = 1:n
            ex = :(A[$i,$j]*B[$j,$j].')
            for k = j + 1:n
                ex = :($ex + A[$i,$k]*B.data[$j,$k].')
            end
            push!(code.args, :($(X[i,j]) = $ex))
        end
    end

    return quote
        @_inline_meta
        @inbounds $code
        return similar_type(A, $T)(tuple($(X...)))
    end
end

@generated function _A_mul_Bt(::Size{sa}, ::Size{sb}, A::StaticMatrix{<:Any,<:Any,Ta}, B::LowerTriangular{Tb,<:StaticMatrix}) where {sa,sb,Ta,Tb}
    m, n = sa[1], sa[2]
    if sb[1] != n
        throw(DimensionMismatch("right hand side B needs first dimension of size $n, has size $(sb[1])"))
    end

    T = promote_matprod(Ta, Tb)
    X = [Symbol("X_$(i)_$(j)") for i = 1:m, j = 1:n]

    code = quote end
    for i = 1:m
        for j = n:-1:1
            ex = :(A[$i,$j]*B[$j,$j].')
            for k = 1:j - 1
                ex = :($ex + A[$i,$k]*B.data[$j,$k].')
            end
            push!(code.args, :($(X[i,j]) = $ex))
        end
    end

    return quote
        @_inline_meta
        @inbounds $code
        return similar_type(A, $T)(tuple($(X...)))
    end
end
