import gymnasium as gym
import tqdm
from DQN.DQN import DQN

def getEnv(env_name="CartPole-v1", is_render=False):
  if is_render:
    return gym.make(env_name, render_mode="human")
  return gym.make(env_name)

def getStepRes(env, action):
  step_res = env.step(action)
  new_observation = step_res[0]
  reward = step_res[1]
  done = step_res[2]
  info = step_res[3]
  return new_observation, reward, done, info

def getReward(observation, new_observation):
  diffAngle = abs(observation[2]) - abs(new_observation[2])
  return diffAngle * 100

def train(env, model, EPISODES, epsilon=0):
  total_rewards = 0
  total_steps = 0
  max_steps = 0

  for episode in tqdm.trange(EPISODES):
    env.reset()
    done = False
    observation = env.unwrapped.state
    steps = 0
    episode_reward = 0
    while not done:
      total_steps += 1
      steps += 1
      if steps > 500:
        break
      action = model.select_action(observation, epsilon=epsilon)
      new_observation, reward, done, info = getStepRes(env, action)
      reward = getReward(observation, new_observation)
      episode_reward += reward
      model.memory.push((observation, action, new_observation, reward))
      observation = new_observation
      model.learn(100)
    model.add_steps_reward(steps, episode_reward)

    total_rewards += episode_reward
    if steps > max_steps:
      max_steps = steps
  if episode % 100 == 0:
    model.update()
  # model.draw(EPISODES)

  average_step = total_steps / EPISODES
  print(f"Average Steps: {average_step},max step:{max_steps}")

def main():
  env = getEnv(is_render="human")
  # env = getEnv()
  dqn = DQN(actions=[0, 1], input_size=4, output_size=2)
  # train(env, dqn, EPISODES=288000)
  train(env, dqn, EPISODES=10, epsilon=0)
  dqn.save_model()
  env.close()

if __name__ == "__main__":
  main()