from collections import deque
import pickle
import random
import os

# 创建记忆回放
class ReplayMemory:
  def __init__(self, capacity):
    self.memory = deque(maxlen=capacity)

  def push(self, transition):
    self.memory.append(transition)

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)
  
  def save_memory(self, filename='memory.pkl'):
    with open(filename, 'wb') as f:
      pickle.dump(self.memory, f)

  def load_memory(self, filename='memory.pkl'):
    if os.path.isfile(filename):
      with open(filename, 'rb') as f:
        self.memory = pickle.load(f)