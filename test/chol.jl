@testset "Cholesky decomposition" begin
    @testset "1×1" begin
        m = @SMatrix [4.0]
        (c,) = chol(m)
        @test c === 2.0
    end

    @testset "2×2" for i = 1:100
        m_a = randn(2,2)
        #non hermitian
        @test_throws ArgumentError chol(SMatrix{2,2}(m_a))
        m_a = m_a*m_a'
        m = SMatrix{2,2}(m_a)
        @test chol(Hermitian(m)) ≈ chol(m_a)
    end

    @testset "3×3" for i = 1:100
        m_a = randn(3,3)
        #non hermitian
        @test_throws ArgumentError chol(SMatrix{3,3}(m_a))
        m_a = m_a*m_a'
        m = SMatrix{3,3}(m_a)
        @test chol(m) ≈ chol(m_a)
        @test chol(Hermitian(m)) ≈ chol(m_a)
    end
    @testset "4×4" for i = 1:100
        m_a = randn(4,4)
        #non hermitian
        @test_throws ArgumentError chol(SMatrix{4,4}(m_a))
        m_a = m_a*m_a'
        m = SMatrix{4,4}(m_a)
        @test chol(m) ≈ chol(m_a)
        @test chol(Hermitian(m)) ≈ chol(m_a)
    end
end

@testset "Cholesky decomposition" begin
    for n = (1, 2, 3, 4),
        m in (SMatrix{n,n}, MMatrix{n,n}),
            elty in (Float32, Float64, BigFloat, Complex64, Complex128)

        A = convert(Matrix{elty}, (elty <: Complex ? complex.(randn(n,n), randn(n,n)) : randn(n,n)) |> t -> t't)
        @test chol(m(A)) ≈ chol(A)
        if elty <: Real
            mCU = cholfact(Symmetric(m(A)))
            CU = cholfact(Symmetric(A))
            @test mCU.uplo == CU.uplo
            @test mCU.factors ≈ CU.factors
            mCL = cholfact(Symmetric(m(A), :L))
            CL = cholfact(Symmetric(A, :L))
            @test mCL.uplo == CL.uplo
            @test mCL.factors ≈ CL.factors
        else
            mCU = cholfact(Hermitian(m(A)))
            CU = cholfact(Hermitian(A))
            @test mCU.uplo == CU.uplo
            @test mCU.factors ≈ CU.factors
            mCL = cholfact(Hermitian(m(A), :L))
            CL = cholfact(Hermitian(A, :L))
            @test mCL.uplo == CL.uplo
            @test mCL.factors ≈ CL.factors
        end
    end
end

@testset "Throw if non-Hermitian" begin
    R = randn(4,4)
    C = complex.(R, R)
    for A in (R, C)
        @test_throws ArgumentError cholfact(A)
        @test_throws ArgumentError chol(A)
    end
end
