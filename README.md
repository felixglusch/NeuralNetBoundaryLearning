# NeuralNetBoundaryLearning
A neural network for boundary learning.

It has ~80% accuracy after training.

I used the [Encog Machine Learning Framework](http://www.heatonresearch.com/encog/) for the neural net.

The network is trained with valid and invalid cartesian coordinates.
Then random coordinates are given and the network guesses whether it is valid or invalid.

A valid coordinate has x, y values >= 0 and <= 10. Invalid coordinates are > 15 and <= 20.
