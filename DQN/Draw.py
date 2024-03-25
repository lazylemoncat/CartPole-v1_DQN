import time
import matplotlib.pyplot as plt
import numpy as np

class Draw:
  def __init__(self):
    self.steps = []
    self.rewards = []
    self.eposides = 0
  
  def add_steps(self, steps):
    self.steps.append(steps)

  def add_reward(self, reward):
    self.rewards.append(reward)

  def drawSteps(self):
    # 画图初始化
    plt.figure(figsize=(100, 50))
    plt.title("Steps")
    # 折线图
    plt.plot(range(self.episodes), self.steps, label="steps")
    # steps平均值
    average_step = sum(self.steps) / len(self.steps)
    plt.axhline(average_step, color='r', linestyle='--', label='Average Steps')
    plt.text(self.episodes/2, average_step, f'Average Steps: {average_step}', ha='center', va='bottom')
    # y轴范围
    min_step = min(self.steps)
    max_step = max(self.steps)
    plt.xlim(0, self.episodes)
    plt.ylim(min_step, max_step)
    y_ticks = np.linspace(min_step, max_step, max_step - min_step + 1)
    plt.yticks(y_ticks)
    # x轴、y轴标签
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    # 图例
    plt.legend()
    # 保存图片
    plt.savefig(f"res/steps{time.strftime('%Y%m%d%H%M%S')}.png")

  def drawRewards(self):
    plt.figure(figsize=(100, 50))
    plt.title("Rewards")
    plt.plot(range(self.episodes), self.rewards, label="reward")
    plt.xlim(0, self.episodes)
    plt.ylim(min(self.rewards), max(self.rewards))
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig(f"res/reward{time.strftime('%Y%m%d%H%M%S')}.png")

  def draw(self, epochs):
    self.episodes = epochs
    self.drawSteps()
    # self.drawRewards()