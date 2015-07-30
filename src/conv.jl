# functions for convolutional layers
type ConvFilter2DLayer
  nr::Int # num rows
  nc::Int # num cols
  ninp::Int # number of input feature maps
  nout::Int # number of output feature maps
  h::Function # non-linear node function
  W::Array{NNMatrix} # matrix of weights
  b::NNMatrix # matrix of biases (actually just a vector)
  wdim::Int
  # function to create a convolutional 2d filter
  function ConvFilter2DLayer(nr::Int, nc::Int, ninp::Int, nout::Int; h=tanh)
    wdim = nr*nc;
    W = Array(NNMatrix, 0);
    for i=1:ninp
      wi = randNNMat(nr, nc);
      wi.w = (wi.w*2.0 - 1.0) ./sqrt(wdim)
      push!(W, wi);
    end

    b = zerosNNMat(nout, 1) #zero vector for biases
    return new(nr, nc, ninp, nout, h, W, b, wdim)
  end
end


function convolve!(g::Graph, A::Array{NNMatrix}, cfilt::ConvFilter2DLayer)
  ninp = length(A)
  if (ninp != cfilt.ninp)
    error("A does not have the required number of inputs")
  end
  if (ninp == 0)
    error("A has zero length")
  end

  # for each input matrix
  # loop through all the possible positions
  nr, nc = size(A[1].w);
  wdim = nr*nc;
  out = Array(NNMatrix, 0);
  for o=1:cfilt.nout
    push!(out, zerosNNMat(nr-cfilt.nr+1, nc-cfilt.nc+1));
  end


  for i=1:ninp, r=1:(nr-cfilt.nr+1), c=1:(nc-cfilt.nc+1)
    # perform a matrix multiply
    temp = cfilt.W[i].w*reshape(A[i].w[r:(r+cfilt.nr-1), c:(c+cfilt.nc-1)], cfilt.wdim,1) + cfilt.b.w
    # put values into the right output feature map
    for o=1:cfilt.nout
      out[o].w[r,c] += temp[o];
    end
  end

  # get our derivatives here
  if g.doBackprop
      push!(g.backprop,
          function ()
              # derivative computation comes here
          end )
  end

  # do our nonlinear transform
  for o=1:cfilt.nout
    out[o] = cfilt.h(g, out[o]);
  end

  return out
end

function pool!(g::Graph, A::NNMatrix, noutx::Int, nouty::Int, f::Function = maximum)
  nx, ny = size(A);
  wx = floor(nx/noutx)
  wy = floor(ny/nouty)

  out = zeros(noutx, nouty)
  sx = 1; sy = 1;
  for i=1:noutx
      sy = 1
      for j=1:nouty
          out[i,j] = f( A[sx:min(sx+wx-1, nx), sy:min(sy+wy-1, ny)])
          sy += wy
      end
      sx += wx
  end
  out

end
