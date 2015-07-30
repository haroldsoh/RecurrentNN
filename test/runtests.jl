#using DRNN
#reload("DRNN.jl")
include("../src/DRNN.jl")
using Base.Test

# graph output test
m1 = DRNN.NNMatrix(3,2)
m1.w[1,1] = 1.; m1.w[1,2] = 2.
m1.w[2,1] = 3.; m1.w[2,2] = 4.
m1.w[3,1] = 5.; m1.w[3,2] = 6.

m2 = DRNN.NNMatrix(2,3)
m2.w[1,1] = 2.; m2.w[1,2] = 3.; m2.w[1,3] = 4.
m2.w[2,1] = 5.; m2.w[2,2] = 6.; m2.w[2,3] = 7.

# add test
g =  DRNN.Graph()
m3 = DRNN.add!(g,m1,m1)
m3.dw[1,1] = .1; m3.dw[1,2] = .2
m3.dw[2,1] = .3; m3.dw[2,2] = .4
m3.dw[3,1] = .5; m3.dw[3,2] = .6

g.backprop[1]()
# set previous chained gradient
@test m1.dw[1,1] == 0.2
@test m1.dw[1,2] == 0.4
@test m1.dw[2,1] == 0.6
@test m1.dw[2,2] == 0.8
@test m1.dw[3,1] == 1.0
@test m1.dw[3,2] == 1.2

# mul test
m1.dw[:] = 0. # reset gradient matrices
m2.dw[:] = 0. # reset  gradient matrices
g =  DRNN.Graph()
m3 = DRNN.mul!(g,m1,m2)
@test m3.w[1,1] == 12.
@test m3.w[1,2] == 15.
@test m3.w[1,3] == 18.
@test m3.w[2,1] == 26.
@test m3.w[2,2] == 33.
@test m3.w[2,3] == 40.
@test m3.w[3,1] == 40.
@test m3.w[3,2] == 51.
@test m3.w[3,3] == 62.

m3.dw[1,1] = .1; m3.dw[1,2] = .2; m3.dw[1,3] = .3
m3.dw[2,1] = .4; m3.dw[2,2] = .5; m3.dw[2,3] = .6
m3.dw[3,1] = .7; m3.dw[3,2] = .8; m3.dw[3,3] = .9
g.backprop[1]()

# m1 gradient tests
@test m1.dw[1,1] == 2.
@test m1.dw[1,2] == 3.8000000000000003
@test m1.dw[2,1] == 4.699999999999999
@test m1.dw[2,2] == 9.2
@test m1.dw[3,1] == 7.4
@test m1.dw[3,2] == 14.600000000000001

# m2 gradient tests
@test m2.dw[1,1] == 4.800000000000001
@test m2.dw[1,2] == 5.7
@test m2.dw[1,3] == 6.6
@test m2.dw[2,1] == 5.999999999999999
@test m2.dw[2,2] == 7.200000000000001
@test m2.dw[2,3] == 8.4


# reul() tests
m4 = DRNN.NNMatrix(3,2)
m4.w[1,1] = 1.; m4.w[1,2] =-2.
m4.w[2,1] =-3.; m4.w[2,2] = 4.
m4.w[3,1] = 5.; m4.w[3,2] =-6.
g =  DRNN.Graph()
m5 = DRNN.relu!(g,m4)
@test m5.w[1,1] == 1.
@test m5.w[1,2] == 0.
@test m5.w[2,1] == 0.
@test m5.w[2,2] == 4.
@test m5.w[3,1] == 5.
@test m5.w[3,2] == 0.

m5.dw[1,1] =-.1; m5.dw[1,2] = .2
m5.dw[2,1] = .3; m5.dw[2,2] = .4
m5.dw[3,1] = .5; m5.dw[3,2] = .6

g.backprop[1]()
@test m4.dw[1,1] == -0.1
@test m4.dw[1,2] == 0.
@test m4.dw[2,1] == 0.
@test m4.dw[2,2] == 0.4
@test m4.dw[3,1] == 0.5
@test m4.dw[3,2] == 0.


# rowpluck!() tests
m4 = DRNN.NNMatrix(3,2)
m4.w[1,1] = 1.; m4.w[1,2] =-2.
m4.w[2,1] =-3.; m4.w[2,2] = 4.
m4.w[3,1] = 5.; m4.w[3,2] =-6.

g =  DRNN.Graph()
m5 = DRNN.rowpluck!(g,m4,2)
@test m5.w[1,1] == -3.
@test m5.w[2,1] == 4.

m5.dw[1,1] =-.1; m5.dw[2,1] = .2
g.backprop[1]()
m4.dw
@test m4.dw[1,1] == 0
@test m4.dw[1,2] == 0
@test m4.dw[2,1] ==-0.1
@test m4.dw[2,2] == 0.2
@test m4.dw[3,1] == 0.
@test m4.dw[3,2] == 0.


# softmax tests
m6 = DRNN.NNMatrix(5,1)
m6.w[1,1] = 0.3; m6.w[2,1] =0.1; m6.w[3,1] =0.6; m6.w[4,1] = 0.002; m6.w[5,1] = 0.00001
sm = DRNN.softmax(m6)
# @test_approx_eq
# @test_approx_eq sm.w[1,1] 0.21494050089813527
# @test_approx_eq sm.w[2,1] 0.17597839816728894
# @test_approx_eq sm.w[3,1] 0.2901393282421457
# @test_approx_eq sm.w[4,1] 0.1595506217827435
# @test_approx_eq sm.w[5,1] 0.15939115090968653


# transpose tests
function transposeTest()
  print("Transpose: ")
  # create a sample matrix
  O = [1.0 2.0 3.0;
       4.0 5.0 6.0;
       7.0 8.0 9.0];
  A = DRNN.NNMatrix(O);
  g = DRNN.Graph();

  # perform a transposition
  out = DRNN.transpose!(g, A);

  # check that the transposition is correct
  @test (out.w == O')

  # perform backprop
  fake_dw = rand(3,3)
  out.dw = fake_dw;
  DRNN.backprop!(g);

  # check that derivatives are correct
  @test (A.dw == out.dw')
  println("PASSED!")
  return true
end

# reshape tests
function reshapeTest()
  print("Reshape: ")
  O = [1.0 2.0 3.0;
       4.0 5.0 6.0;
       7.0 8.0 9.0];
  A = DRNN.NNMatrix(O);
  g = DRNN.Graph();

  # perform a transposition
  out = DRNN.reshape!(g, A, 9, 1);

  # check that the transposition is correct
  @test (out.w == reshape(O, 9, 1))

  # perform backprop
  fake_dw = rand(3,3)
  out.dw = fake_dw;
  DRNN.backprop!(g);

  # check that derivatives are correct
  @test (A.dw == reshape(out.dw, 3, 3))
  println("PASSED!")
end

# reshape tests
function sumTest()
  print("Sum: ")
  O = [1.0 2.0 3.0;
       4.0 5.0 6.0;
       7.0 8.0 9.0];
  A = DRNN.NNMatrix(O);
  g = DRNN.Graph();

  # perform a transposition
  out = DRNN.sum!(g, A);

  # check that the transposition is correct
  @test (out.w[1] == 45.0)
  @test size(out.w) == (1,1)
  # perform backprop
  out.dw[1] = 3.0;
  DRNN.backprop!(g);

  # check that derivatives are correct
  @test (A.dw[1] == 3.0)
  println("PASSED!")
end

function convTest()
  print("Convolution: ")
  # create sample data from known working functions
  include("genConvTestCase.jl");

  # create a convolutional filter
  nr = fnx;
  nc = fny;
  conv = DRNN.ConvFilter2DLayer(fnx, fny, ninp, nout, h=DRNN.tanh!)

  # transfer model
  st = 1;
  en = st+(nr*nc-1);
  for i=1:ninp
    conv.W[i].w = cfilt.W[:,st:en];
    st = en+1;
    en = st+(nr*nc-1);
  end

  for o=1:nout
    conv.b.w[o] = cfilt.b[o];
  end

  # transfer input
  Amat = Array(DRNN.NNMatrix, 0)
  for i=1:ninp
    push!(Amat, DRNN.NNMatrix(A[:,:,i]));
  end

  # perform convolution
  g = DRNN.Graph()
  tout = DRNN.convolve!(g, Amat, conv)

  # compare convolutional test to output
  for o=1:nout
      for r=1:size(out,1), c=1:size(out,2)
        @test_approx_eq out[r,c,o] tout[o].w[r,c]
      end
  end

  # TODO: Derivative tests come here

  println("PASSED!");
end

# run tests
transposeTest()
reshapeTest()
sumTest()
convTest()
