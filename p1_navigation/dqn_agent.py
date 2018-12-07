# adapted from https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Dueling%20Double%20DQN%20with%20PER%20and%20fixed-q%20targets/Dueling%20Deep%20Q%20Learning%20with%20Doom%20%28%2B%20double%20DQNs%20and%20Prioritized%20Experience%20Replay%29.ipynb

import numpy as np
import random
from collections import namedtuple,deque

import torch
from torch import nn,optim
import torch.nn.functional as F

from SumTree import SumTree
from model import QNetwork

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lr = 5e-4
batch_size = 64
capacity  = int(1e5)
tau = 1e-3
gamma = 0.99
UPDATE_EVERY = 4

class Agent:
    def __init__(self,state_size,action_size,seed,prioritized=False,dueling=False):
        
        self.prioritized = prioritized
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        self.qnetwork_local = QNetwork(state_size,action_size,seed,dueling=dueling)
        self.qnetwork_target = QNetwork(state_size,action_size,seed,dueling=dueling)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),lr=lr)
        
        if prioritized == True:
            self.memory = PRIOREPLAYBUFFER(capacity,batch_size,action_size,seed)
        else:
            self.memory = REPLAYBUFFER(capacity,batch_size,action_size,seed)
            
        self.t_step = 0
    
    def step(self,state,action,reward,next_state,done):
        self.memory.add(state,action,reward,next_state,done)
        
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if self.prioritized:
                if self.memory.memory.n_entries > batch_size:
                    idxs,weights,experiences = self.memory.sample()
                    self.learn(experiences,gamma,idxs,weights)
            else:
                if len(self.memory) > batch_size:
                    experiences = self.memory.sample()
                    self.learn(experiences, gamma,None,None)
    
    def act(self,state,eps):
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def learn(self,experiences,gamma,idxs,weights):
        
        states,actions,rewards,next_states,dones = experiences
        action_idxs = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        
        Q_tar_vals = self.qnetwork_target(next_states).detach().gather(1,action_idxs)
        
        Q_tar = rewards + ((1-dones)*gamma*Q_tar_vals)
        
        Q_est = self.qnetwork_local(states).gather(1,actions)
        
        if self.prioritized == True:
            errors = F.l1_loss(Q_est,Q_tar,reduce=False)
        
            self.memory.batch_updates(idxs,errors.cpu().data.numpy())
        
            weights = torch.from_numpy(weights).float().to(device)
            loss = (weights * F.smooth_l1_loss(Q_est,Q_tar,reduce=False)).mean()
            
        else:
            loss = F.mse_loss(Q_est,Q_tar)
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.soft_update(self.qnetwork_local,self.qnetwork_target,tau)
    
    def soft_update(self,local_n,target_n,tau):
        for lp,tp in zip(local_n.parameters(),target_n.parameters()):
            tp.data.copy_(lp.data*tau + (1-tau)*tp.data)
    

    
class PRIOREPLAYBUFFER:
    
    alpha = 0.6
    beta = 0.4
    beta_update = 0.001
    e = 0.01
    
    def __init__(self,capacity,batch_size,action_size,seed):
        
        self.batch_size = batch_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.experience = namedtuple('Experience',field_names=['state','action','reward','next_state','done'])
        self.memory = SumTree(capacity)
    
    def add(self,state,action,reward,next_state,done):
        
        e = self.experience(state,action,reward,next_state,done)
        
        max_priority = np.max(self.memory.tree[-self.memory.capacity:])
        if max_priority == 0:
            max_priority = 1
            
        self.memory.add(max_priority,e)
    
    def sample(self):
        idxs = []
        experiences = []
        priorities = []
        
        seg = self.memory.total_priority() / self.batch_size
        self.beta  = np.min([1,self.beta + self.beta_update])
        
        # min of all not just sampled probs
        prob_min = np.min(self.memory.tree[self.memory.capacity-1:self.memory.capacity-1+self.memory.n_entries]) / self.memory.total_priority()
       
        
        for i in range(self.batch_size):
            a = seg * i
            b = seg * (i+1)
            val = np.random.uniform(a,b)
            
            idx,pri,exp = self.memory.get_value(val)
            
            idxs.append(idx)
            experiences.append(exp)
            priorities.append(pri)
        
        probabilities = np.array(priorities) / self.memory.total_priority()
        weights = np.power(probabilities / prob_min,-self.beta)
        
        states = torch.from_numpy(np.vstack(e.state for e in experiences if e is not None)).float().to(device)
        actions = torch.from_numpy(np.vstack(e.action for e in experiences if e is not None)).long().to(device)
        rewards = torch.from_numpy(np.vstack(e.reward for e in experiences if e is not None)).float().to(device)
        next_states = torch.from_numpy(np.vstack(e.next_state for e in experiences if e is not None)).float().to(device)
        dones = torch.from_numpy(np.vstack(e.done for e in experiences if e is not None).astype(np.uint8)).float().to(device)
        
        return np.array(idxs),np.array(weights,dtype=np.float32),(states,actions,rewards,next_states,dones)
    
    def batch_updates(self,idxs,errors):
        errors += self.e
        clip_errors = np.minimum(1.0,errors)
        ps = np.power(clip_errors,self.alpha)
        
        for idx,p in zip(idxs,ps):
            self.memory.update(idx,p)
                      
class REPLAYBUFFER:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size,action_size,seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)