abstract Model # this is either an LSTM or RNN

type NNMatrix # Neural net layer's weights & gradients
    n::Int
    d::Int
    w::Matrix{Float64} # matrix of weights
    dw::Matrix{Float64} # matrix of gradients
    NNMatrix(n::Int) = new(n, 1, zeros(n,1), zeros(n,1))
    NNMatrix(n::Int, d::Int) = new(n, d, zeros(n,d), zeros(n,d))
    NNMatrix(n::Int, d::Int, w::Matrix{Float64}, dw::Matrix{Float64}) = new(n, d, w, dw)
    NNMatrix(w::Array) = new(size(w,1), size(w,2), w, zeros(size(w,1),size(w,2)))
    function NNMatrix(w::Matrix{Float64}, dw::Matrix{Float64})
      if size(w) != size(dw)
        error("Matrix and Gradient Matrix sizes do not match");
      end
      new(size(w,1), size(w,2), w, dw)
    end
end

randNNMat(n::Int, d::Int, std::FloatingPoint=1.) = NNMatrix(n, d, randn(n,d)*std, zeros(n,d))
onesNNMat(n::Int, d::Int) = NNMatrix(n, d, ones(n, d), zeros(n, d))
zerosNNMat(n::Int, d::Int) = NNMatrix(n,d)

function softmax(m::NNMatrix)
    out = NNMatrix(m.n,m.d)
    maxval = maximum(m.w)
    out.w[:] = exp(m.w - maxval)
    out.w /= sum(out.w)
    return out
end
