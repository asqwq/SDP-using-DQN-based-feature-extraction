import pandas as pd
import numpy as np
import random
from collections import deque
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split

# 加载数据集
data = pd.read_csv('D:\DQN-SDP\Dataset\pc1.csv')

# 将数据集分为训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# 对训练集和测试集进行归一化处理
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data.drop('defects', axis=1))
test_data_scaled = scaler.transform(test_data.drop('defects', axis=1))

# 定义超参数
num_epochs = 1000
batch_size = 32
gamma = 0.99
initial_epsilon = 1.0
final_epsilon = 0.01
epsilon_decay = 0.995
capacity = 10000
input_dim = train_data_scaled.shape[1]
num_actions = 2


# 定义DQN模型
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 定义损失函数和优化器
model = DQN(input_dim, num_actions)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 定义经验回放缓冲区类
class ReplayBuffer():
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# 定义经验回放缓冲区
replay_buffer = ReplayBuffer(capacity)


# 定义选择动作的方法
def select_action(state, epsilon):
    if random.random() < epsilon:
        action = random.randint(0, num_actions - 1)
    else:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_value = model(state)
        action = q_value.argmax().item()
    return action


# 定义训练函数
def train(model, optimizer, replay_buffer, batch_size, gamma):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    state = torch.FloatTensor(state).to(device)
    action = torch.LongTensor(action).to(device)
    reward = torch.FloatTensor(reward).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    done = torch.FloatTensor(done).to(device)

    q_value = model(state).gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = model(next_state).max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = criterion(q_value, expected_q_value.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epsilon = initial_epsilon
train_rewards = []
test_rewards = []

for epoch in range(num_epochs):
    state = train_data_scaled[0]
    done = False
    total_reward = 0

while not done:
    action = select_action(state, epsilon)
    next_state = train_data_scaled[len(train_rewards)+1]
    reward = train_data.iloc[len(train_rewards)]['defects']
    done = len(train_rewards) >= len(train_data)-2

    replay_buffer.push(state, action, reward, next_state, done)
    state = next_state
    total_reward += reward

    if len(replay_buffer) > batch_size:
        train(model, optimizer, replay_buffer, batch_size, gamma)

train_rewards.append(total_reward)
epsilon = max(epsilon * epsilon_decay, final_epsilon)

# 在测试集上测试模型
test_state = test_data_scaled[0]
test_done = False
test_total_reward = 0

while not test_done:
    test_action = select_action(test_state, 0)
    test_next_state = test_data_scaled[len(test_rewards)+1]
    test_reward = test_data.iloc[len(test_rewards)]['defects']
    test_done = len(test_rewards) >= len(test_data)-2

    test_state = test_next_state
    test_total_reward += test_reward

test_rewards.append(test_total_reward)

# 打印训练结果
if (epoch+1) % 10 == 0:
    print(f"Epoch: {epoch+1}/{num_epochs}, Train Reward: {train_rewards[-1]:.2f}, Test Reward: {test_rewards[-1]:.2f}, Epsilon: {epsilon:.2f}")


