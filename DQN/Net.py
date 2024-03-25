import torch
import torch.nn as nn

# 定义神经网络模型
class Net(nn.Module):
  # 输入一个1*16的张量，输出一个1*4的张量
  def __init__(self, input_size, output_size):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(input_size, 8)
    self.fc2 = nn.Linear(8, 16)
    self.fc3 = nn.Linear(16, 8)
    self.fc4 = nn.Linear(8, 4)
    self.fc5 = nn.Linear(4, output_size)

  def forward(self, x):
    leaky_relu = nn.LeakyReLU()
    x = leaky_relu(self.fc1(x))
    x = leaky_relu(self.fc2(x))
    x = leaky_relu(self.fc3(x))
    x = leaky_relu(self.fc4(x))
    x = self.fc5(x)
    return x
