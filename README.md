# Making a Deep Q Learning model for pong
Short learning project, mostly learning how to do neural network training
We are implementing this paper:://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
(This has now been modified to not do random batches anymore as it will take very long for every play)

Currently should run a 
xt ∈ R^d each of this is an image
Pseudocode of DQL with experience replay:
```
Initialize replay memory D to capacity N
Initialize action-value function Q with random weights
for episode = 1, M do -> This will be train
	Initialise sequence s_1 = {x_1} and preprocessed sequenced φ_1 = φ(s1)
	for t = 1, T do -> this is playing a game on the model
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
		- environment
	functions:
		- train: 
			Pseudocode: 
			-
				memory <- empty
				model <- untrained model
				for each episode:
					current-state = environment's initial state
					processed-state = process-state(state)
					for each step at playing the game or it timeouts:
						action <- make epsilon-greedy decision based on the current state
						get reward, next state, is_terminal from the actions on the environment
						store this transition (s_{t}, action, s_{t + 1), is_terminal) into memory
						batch <- sample a random batch (might consider batch size here)
						rewards = empty
						for each sample in the batch:
							label each sample's reward with
							reward if terminal state
							reward = reward + gamma * model's prediction on next state
							add reward to rewards
						train the model on the batch's states and rewards
		- replay
		- save_weights
		- epsilon_greedy
			- make a random move with probability y
