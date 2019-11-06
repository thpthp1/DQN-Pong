# Making a Deep Q Learning model for pong

We are implementing this paper:://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

   

Class architecture

Agent:
	objects:
		- Model (tf.model)
		- gamma
		- state-memory (either to a memory list/stack or a text file)
		- Episodes
	functions:
		- train
		- replay
		- save_weights
		- play
