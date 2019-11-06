# Making a Deep Q Learning model for pong

We are implementing this paper:://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

xt ∈ R^d each of this is an image
Pseudocode of DQL with experience replay:
```
Initialize replay memory D to capacity N
Initialize action-value function Q with random weights
for episode = 1, M do
	Initialise sequence s_1 = {x_1} and preprocessed sequenced φ_1 = φ(s1)
	for t = 1, T do
		With probability  select a random action at
		otherwise select at = maxa Q∗
		(φ(st), a; θ)
		Execute action at in emulator and observe reward rt and image xt+1
		Set s_{t+1} = s_t, a_t, x_{t+1} and preprocess φt+1 = φ(st+1)
		Store transition (φ_t, a_t, rt, φ_{t+1}) in D
		Sample random minibatch of transitions (φ_j , aj , rj , φ_{j+1}) from D
		Set yj =
		
		rj for terminal φ_{j+1}
		rj + γ maxa0 Q(φ_{j+1}, a_0; θ) for non-terminal φj+1
		Perform a gradient descent step on (yj − Q(φ_j , a_j ; θ))2
		according to equation 3
	end for
end for
```
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
