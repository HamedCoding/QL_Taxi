# Q Learning / TAXI-V3
import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3")

action_size=env.action_space.n
state_size=env.observation_space.n
q=np.zeros((state_size, action_size))

train_episodes=40_000
test_episodes=100
steps=500
gamma=0.9
lr=0.9

epsilon=1
max_epsilon=1
min_epsilon=0.01

decay=0.0001

# train
train_rewards=[]
epsilon_lst=[]
for episode in range(train_episodes):
    state=env.reset()
    tot_reward=0
    done=False
    print('******* EPISODE ', episode, ' *******')
    for step in range(steps):
        rand_epsilon=random.uniform(0,1)

        if rand_epsilon>epsilon:
            action=np.argmax(q[state,])
        else:
            action=random.randint(0,3)

        new_state, reward, done, info = env.step(action)

        q[state,action]=q[state,action]+lr*(reward+ gamma* np.max(q[new_state,])-q[state,action])

        state = new_state
        tot_reward+=reward

        if done:
            break

    train_rewards.append(tot_reward)
    print(tot_reward)
    epsilon=(max_epsilon - min_epsilon) * np.exp(-decay*episode) + min_epsilon
    epsilon_lst.append(epsilon)
    print(round(epsilon,2))

print(' Train - mean reward= ', round(np.mean(train_rewards),1))

# test
env.reset()
test_rewards=[]
test_done=[]
test_steps=[]

for episode in range(test_episodes):
    state = env.reset()
    done = False
    tot_reward=0
    step_no=0
    print('******* EPISODE ',episode, ' *******')

    for step in range(steps):
        action = np.argmax(q[state,])
        new_state, reward, done, info = env.step(action)

        q[state, action] = q[state, action] + lr * (reward + gamma * np.max(q[new_state,]) - q[state, action])

        state = new_state
        tot_reward+=reward
        step_no=step

        if episode==0:
            env.render()

        if done:
            print(tot_reward)
            break

    test_rewards.append(tot_reward)
    test_steps.append(step_no)
    if done==True:
        test_done.append(1)
    else:
        test_done.append(0)

env.close()

print(' Test - mean reward= ', int(np.mean(test_rewards)))

fig=plt.figure(figsize=(10,12))
plt.subplot(511)
plt.plot(list(range(len(train_rewards))), train_rewards)
plt.plot(list(range(len(train_rewards))), len(train_rewards)*[np.mean(train_rewards)])
plt.title('Taxi-v3 Results\n\nTotal train reward in each episode')
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.legend(['Train reward','Mean reward'],loc='upper left')

plt.subplot(512)
plt.plot(list(range(len(epsilon_lst))), epsilon_lst)
plt.title('Epsilon')
plt.ylabel('Epsilon')
plt.xlabel('Episode')

plt.subplot(513)
plt.plot(list(range(len(test_rewards))), test_rewards)
plt.title('Total test reward in each episode')
plt.ylabel('Reward')
plt.xlabel('Episode')

plt.subplot(514)
plt.plot(list(range(len(test_steps))), test_steps)
plt.title('Total test steps in each episode')
plt.ylabel('Steps')
plt.xlabel('Episode')

plt.subplot(515)
plt.plot(list(range(len(test_done))), test_done)
plt.title('Win(1) or Lose(0)')
plt.xlabel('Episode')
plt.yticks([0,1])

plt.savefig('result.png',dpi=300)
plt.show()
