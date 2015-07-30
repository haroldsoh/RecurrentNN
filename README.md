# DRNN.jl

DRNN.jl is a fork of the [RecurrentNN.jl](https://github.com/Andy-P/RecurrentNN.jl) package by [Andre Pemmelaar](https://github.com/Andy-P), which is a Julia language package based on Andrej Karpathy's [RecurrentJS](http://cs.stanford.edu/people/karpathy/recurrentjs) library in javascript.
It implements:

- Deep **Recurrent Neural Networks** (RNN)
- **Long Short-Term Memory networks** (LSTM)
- **Gated Recurrent Neural Networks** (GRU)
- **Gated Feedback Recurrent Neural Networks** (GF-RNN)
- **Gated Feedback Long Short-Term Memory networks** (GF-LSTM)

The library can construct arbitrary **expression graphs** allowing **automatic differentiation** similar to what you may find in Theano for Python, or in Torch. Currently, the code uses this very general functionality to implement RNN/LSTM/GRU, but one can build arbitrary Neural Networks and do automatic backprop.

For information an the **Gated Feedback** variants see [Gated Feedback Recurrent Neural Networks](http://arxiv.org/abs/1502.02367)


## Example code for LSTM

To construct and train an LSTM for example, you would proceed as follows:

```julia
using DRNN

# takes as input Mat of 10x1, contains 2 hidden layers of
# 20 neurons each, and outputs a Mat of size 2x1
hiddensizes = [20, 20]
outputsize = 2
cost = 0.
lstm = LSTM(10, hiddensizes, outputsize)
x1 = randNNMat(10, 1) # example input #1
x2 = randNNMat(10, 1) # example input #2
x3 = randNNMat(10, 1) # example input #3

# pass 3 examples through the LSTM
G = Graph()
# build container to hold output after each time step
prevhd   = Array(NNMatrix,0) # holds final hidden layer of the recurrent model
prevcell = Array(NNMatrix,0) #  holds final cell output of the LSTM model
out  = NNMatrix(outputsize,1) # output of the recurrent model
prev = (prevhd, prevcell, out)

out1 = forwardprop!(G, lstm, x1, prev)
out2 = forwardprop!(G, lstm, x2, out1);
out3 = forwardprop!(G, lstm, x3, out2);

# the last part of the tuple contains the outputs:
outMat =  prev[end]

# for example lets assume we have binary classification problem
# so the output of the LSTM are the log probabilities of the
# two classes. Lets first get the probabilities:
probs = softmax(outMat)
ix_target = 1 # suppose first input has target class

# cross-entropy loss for softmax is simply the probabilities:
outMat.dw = probs.w
# but the correct class gets an extra -1:
outMat.dw[ix_target] -= 1;

# in real application you'd probably have a desired class
# for every input, so you'd iteratively see the .dw loss on each
# one. In the example provided demo we are
# predicting the index of the next letter in an input sentence.

# update the LSTM parameters
backprop!(G)
s = RMSPropSolver() # RMSProp optimizer

# perform RMSprop update with
# step size of 0.01
# L2 regularization of 0.00001
# and clipping the gradients at 5.0 elementwise
sparams = RMSPropSolverParams(0.01, 0.0001, 5.0)
step!(s, lstm, sparams);
```


## Character Sequence Memorization Example  

An demo that memorizes character sequences (based on the [recurrentjs demo](http://cs.stanford.edu/people/karpathy/recurrentjs) can be found in [example/example.jl](https://github.com/Andy-P/DRNN.jl/blob/master/example/example.jl). In brief, the Deep RNN learns how to construct sentences at the character level by learning from sample sentences.

##Credits
This library draws on the work of [Andrej Karpathy](https://github.com/karpathy). Speed enhancements were added by [Iain Dunning](https://github.com/IainNZ). The Gated Recurrent Neural Network implementation and Gated Feedback variants were added by [Paul Heideman](https://github.com/paulheideman). [Harold Soh](https://github.com/haroldsoh) implemented the Adam optimizer.

## FAQ
#### Why the fork and change of name?
I wanted to implement some very experimental functions (attention networks). I found "RecurrentNN" too long a name and thought "DRNN" for "Deep Recurrent Neural Nets" to be more appropriate.

#### Future plans?
This library may be pulled back into the main RecurrentNN library in the future.

## License
MIT
