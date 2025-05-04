- final report is a pdf file
	- we kinda just put whatever we want + plots below
- there will be a demo
- due may 5th, 11:55pm
- useful links:
	- http://inst.eecs.berkeley.edu/~cs188/sp11/projects/classification/classification.html - doesn't work without a berkeley login
	- http://rl.cs.rutgers.edu/fall2019/data.zip - data
- data stuff/testing modules (split using this):
  	- faces data - rajeev
  	- digits data - dhvani
  	- testing module - jasmin
- we need to implement 3 models (also split using this):
	- perceptron - dhvani
	- 3-layer NN (input, hidden 1, hidden 2, output) from scratch - rajeev
	- 3-layer NN (input, hidden 1, hidden 2, output) using PyTorch - jasmin
- train with 10% - 100% (increments of 10) of the data

- plot
	- time needed for training vs number of data points used
	- prediction error (test) + stdev as vs number of data points used
- can't use PyTorch unless specifically asked to do so
- 70% accuracy is "good enough"
- design our own features - simpler the better - each pixel could be a feature
- standard deviation computation
- 	- choose random seed
	- suppose we're doing n% of the data
	- use a different random sample n% for 5 iterations of training
	- then combine

## digit data formatting
- each line is 29 characters long
- ends with a newline character, regardless of emptiness

## face data formatting
- add notes here
