import numpy as np
import torch
from torch import nn,optim
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self,state_size,action_size,seed,dueling=False):
        super(QNetwork,self).__init__()
        
        self.dueling = dueling
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)
        
        fc1_units  = 128
        fc2_units = 64
        
        self.fc1 = nn.Linear(state_size,fc1_units)
        
        if dueling:
            self.fc2_val = nn.Linear(fc1_units,fc2_units)
            self.fc2_adv = nn.Linear(fc1_units,fc2_units)
        
            self.fc3_val = nn.Linear(fc2_units,1)
            self.fc3_adv = nn.Linear(fc2_units,action_size)
            
        else:
            
            self.fc2 = nn.Linear(fc1_units,fc2_units)
            self.fc3 = nn.Linear(fc2_units,action_size)
        
    def forward(self,state):
        
        if self.dueling == False:
            
            state = F.relu(self.fc1(state))
            state = F.relu(self.fc2(state))
            
            return self.fc3(state)
        
        else:
            
            state = F.relu(self.fc1(state))
        
            val = F.relu(self.fc2_val(state)) 
            adv = F.relu(self.fc2_adv(state))
        
            val = self.fc3_val(val)
            adv = self.fc3_adv(adv)
        
            return val + adv - adv.mean()
            