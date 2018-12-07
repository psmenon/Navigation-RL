[//]: # (Image References)

[image]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Navigation-RL

## Introduction

In this project I trained an agent to navigate and collect as many yellow bananas as possible while avoiding blue bananas in a large, square world. A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.

![Trained Agent][image]

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.

The action space is discrete and the agent can take 4 possible actions:

- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic and is considered solved when the agent gets an averge score of +13 over 100 consecutive episodes.

## Training in Linux

1. Download the environment:  

    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    
To train the agent on Amazon Web Services (AWS), and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

Create and activate a Python 3.6 environment using Anaconda (name_of_environment can be replaced with any name):
   
   	conda create --name name_of_environment python=3.6
	source activate name_of_environment

