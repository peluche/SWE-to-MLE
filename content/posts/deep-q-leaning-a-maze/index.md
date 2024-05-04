---
title: "DQN: Deep Q-Leaning a Maze"
date: 2023-10-06T18:44:29+02:00
draft: false
author: "peluche"
authorLink: "https://github.com/peluche"
description: "DQN: Deep Q-Leaning a Maze"
images:
- "posts/deep-q-leaning-a-maze/minotaur.png"
tags: ['DQN', 'RL', 'pytorch', 'maze']
categories: ['bestiary']
resources:
- name: "minotaur"
  src: "minotaur.png"
- name: "policy"
  src: "policy.png"
- name: "policy2"
  src: "policy2.png"
math:
  enable: true
---

Adding a new entry to the bestiary, the Minotaur.

![minotaur](minotaur.png 'Minotaur by stable diffusion')

## The Quest
As a first step toward Reinforcement Learning (RL) let's write a maze solver using Deep Q-Network (DQN).

## Bellman's Equation
To me DQN seems to be the RL technique requiring the least effort. All you need to do is to balance the left side of the Bellman's equation with its right side:

$$Q(s, a) = R + \gamma . max_i(Q(s', a_i))$$

For our purpose `Q()` is the neural network. `s` (aka. state) and `a` (aka. action) are the input of the network, here it would be the maze and the current position. `R` is the reward for taking action `a` (i.e. hitting a wall is a `-1`, keeping on the path is a `0` and finding the exit is a `10`). $\gamma$ (gamma) (aka. decay rate) is how much we discount future rewards.

And all we want, is for our network to be consistent by predicting that the expected value of being at position `s` and taking the action `a` is the same as having already done action `a` and having been rewarded for it if we keep playing optimally afterward.

## Implementation
Because we are only trying to balance the Bellman's equation we don't need any extra cleverness. We only need possible positions to look at and evaluate (even if the network weights are totally random at initialization) and let the magic of gradient descent narrow down a consistent `Q()` for us.

Define the neural network

```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, len(MOVES)),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)        
        return logits
```

I did several experiments with larger hidden layers, or deeper networks and found out that this simple tiny one was outperforming what I could get with much bigger ones. My guess is that the extra complexity was slowing down the training more than it was actually contributing to the quality of the answer.

Here we have several options for generating our training set:
- self play, we start at the entrance of the maze and move around according to some exploration rate
- random position, teleport somewhere in the maze and do a single move
- exhaustive play, teleport everywhere in the maze and try every move

It feels like self play is the more realistic option for writing a Go engine, but for the purpose of outsmarting the Minotaur I went with exhaustive play. It lets me train much faster by batching a lot of positions and move together in one big vectorized pass of the network, instead of doing moves one by one.

```python
def get_next_pos(maze, rewards, pos, move):
    is_terminal = True # default to a terminal state.
    new_pos = pos # default to forbidden move.
    reward = HIT_WALL_PENALTY # default to hitting a wall.
    x, y = pos
    a, b = maze.shape
    i, j = move
    if 0 <= x + i < a and 0 <= y + j < b:
        new_pos = (x + i, y + j)
        reward = get_reward(rewards, new_pos)
        is_terminal = maze[new_pos] != 1
    return new_pos, reward, move, is_terminal

def get_batch_exhaustive_search():
    batch = []
    maze, rewards = get_maze()
    for pos in (maze == 1).nonzero().tolist():
        for mm in list(MOVES.keys()):
            new_pos, reward, move, is_terminal = get_next_pos(maze, rewards, pos, mm)
            batch.append((pos, move, new_pos, reward, is_terminal))
    return maze, batch
```

And now we just train long enough for Q to get stable. Have a look at what direction the network predict for each position at different training steps.

![policy](policy.png 'Predictions for the exit')

Let's compare the distances to the exit as computed by BFS with the policy's predicted reward at each position.

![policy2](policy2.png 'BFS Distance vs Predicted Reward')

## Extra curicular activities
Here's a random bunch of things that could be used to improve the code:
- experience replay, if we implement self play it will train slowly, one option is to save the states we encounter and replay them in a batch.
- target network, we can fix the `Q()` on the right side of the Bellman's equation to a set of weight, run a bunch of training and only then update it with our new Q. This makes the training more stable. This also happen to be somewhat emulated by just running bigger batches so I went with that instead.
- convolution, instead of flattening the maze into a 1d vector and feeding everything throug linear layers, we could feed a 2d matrix maze into a bunch of conv2d layers.
- one_hot vs raw coordinates, I went with encoding positions in the maze as two one_hot encoded vectors, but another approach would be to feed the X and Y coordinate as a flaot and see what happen.

## The code
You can get the code at [https://github.com/peluche/rl/](https://github.com/peluche/rl/)

{{< gist peluche feba2e545a469aca8aceb85fcdbdacab >}}

## Sources
The maze's walls were full of obscure writings, forbidden knowledge on this arcane spell left by Bellman eons ago, some of these writing are transcibed here:
- https://www.analyticsvidhya.com/blog/2019/04/introduction-deep-q-learning-python/
- https://thenerdshow.com/maze.html
- https://tomroth.com.au/dqn-nnet/
- https://ai.stackexchange.com/questions/35184/how-do-i-design-the-network-for-deep-q-network
- https://web.stanford.edu/class/cs234/CS234Win2019/slides/lnotes6.pdf

And a special thanks to [https://github.com/changlinli](https://github.com/changlinli) and his awesome set of lectures at the [Recurse Center](https://www.recurse.com/scout/click?t=dcdcd5fced9bfab4a02b4dd6bb05199e) for inspiring me to work on this.
