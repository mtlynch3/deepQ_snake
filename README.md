# deepQ_snake
To run: python3 snake.py

Made using pytorch and pygame.

The agent (snake) learns to play the game via Deep Q-learning. A type of reinforcement learning, a Q-learning agent will recieve awards from the environment (game) based on its actions. It will then use this reward to update its policy (how it makes decisions ie chooses the next move). The policy is a 2-layer neural network: the inputs are 11 boolean parameters, a hidden layer of size 256, and an output layer of size 3. The inputs to the network are: immediate danger straight ahead, to the left or right; the direction the snake is currently moving, up, down, left, or right; the direction of fruit in relation to the snakes position, above, below, left, or right. The network ouputs the probabilities of maximum reward for all 3 directions. The maximum of the 3 probabilities (the action that maximizes reward) is chosen.
