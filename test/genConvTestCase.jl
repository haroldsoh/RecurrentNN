
# working convolution functions
type ConvFilter2D
    W::Array{Float64}
    b::Array{Float64}
    h::Function
    fnx::Int
    fny::Int
end

function createConvFilter(fnx::Int, fny::Int, ninp::Int, nout::Int; h=tanh)
    wdim = fnx*fny*ninp;
    W = rand(nout, wdim)*2 - 1; #random filter
    W = W./sqrt(wdim)
    b = zeros(nout);

    return ConvFilter2D(W, b, h, fnx, fny)
end


function convolve(A::Array, cfilt::ConvFilter2D)
    nx,ny,ni = size(A)
    fnx = cfilt.fnx
    fny = cfilt.fny
    nout = size(cfilt.W,1)
    out = zeros(nx-fnx+1, ny-fny+1, nout)

    for i=1:(nx-fnx+1)
        for j=1:(ny-fny+1)
            C = h(cfilt.W*(A[i:(i+fnx-1), j:(j+fny-1), :])[:] + cfilt.b)
            out[i,j,:] = reshape(C, 1, 1, nout)
        end
    end
    return out
end


fnx = 3; #rows
fny = 3; #cols
ninp = 5;
nout = 2;
h = tanh

# create some random data to convolve upon
A = rand(8, 8, ninp)

# create the convolutional filter
cfilt = createConvFilter(fnx, fny, ninp, nout, h=tanh)

# perform the convolution
out = convolve(A, cfilt)

# save the data
using HDF5, JLD
save_unit_test_data = true
if save_unit_test_data
    save("conv_unit_test_data.jld", "input_data", A,
    "out", out ,
    "conv_weight", cfilt.W,
    "conv_bias", cfilt.b,
    "conv_filter_size", [fnx fny]);
end
