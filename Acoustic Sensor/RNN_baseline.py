import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import LambdaLR


torch.manual_seed(0)


class FullyConnectedNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FullyConnectedNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, Fs):
        ypre = []
        x = torch.tensor([0.0, 0.0, 0.0, 0.0])

        for i in range(len(Fs)):
            x_next = self.network(torch.cat([x, Fs[i:i+1]], dim=0).float())
            ypre.append(x_next)
            x = x_next

        return torch.stack(ypre)


def RMSE_VAF(y, y_hat):
    RMSE = np.linalg.norm(y-y_hat)/np.sqrt(len(y))
    VAF = 1-np.linalg.norm(y-y_hat)**2/np.linalg.norm(y)**2
    return RMSE, VAF


def rule(epoch):
    if epoch < warm_up_step:
        return epoch/warm_up_step
    else:
        return pow(gamma, int((epoch-warm_up_step)/25))


y_num = 500
y_now = 100
data = np.load('data/ChirpStimulation.npy')
Fs_signal = torch.tensor(data[0:y_now, 0])
y_output = torch.tensor(data[0:y_now, 1])
c_output = torch.tensor(data[0:y_now, 2])


input_size = 5
hidden_size = 64
output_size = 4
learning_rate = 0.0001
warm_up_step = 8000
gamma = 0.98

model = FullyConnectedNet(input_size, hidden_size, output_size)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = LambdaLR(optimizer, lr_lambda=rule)

num_epochs = 15000
tic = 0
for epoch in range(num_epochs):

    if y_now > 100:
        tic += 1

    y_pre = model(Fs_signal)
    loss = (y_pre[:, 0] - y_output).pow(2).sum() + (y_pre[:, 1] - c_output).pow(2).sum()

    if loss <= 1:
        if y_now < y_num:
            tic = 0
            y_now += 10
            Fs_signal = torch.tensor(data[0:y_now, 0])
            y_output = torch.tensor(data[0:y_now, 1])
            c_output = torch.tensor(data[0:y_now, 2])

            if y_now % 100 == 0:
                # change learning rate
                for param_group in optimizer.param_groups:
                    learning_rate = learning_rate / 2.5
                    param_group['lr'] = learning_rate
        else:
            break

    if tic == 250:
        if y_now < y_num:
            tic = 0
            y_now += 10
            Fs_signal = torch.tensor(data[0:y_now, 0])
            y_output = torch.tensor(data[0:y_now, 1])
            c_output = torch.tensor(data[0:y_now, 2])

            if y_now % 100 == 0:
                # change learning rate
                for param_group in optimizer.param_groups:
                    learning_rate = learning_rate / 2.5
                    param_group['lr'] = learning_rate

        else:
            break

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    scheduler.step()

    if epoch % 100 == 0:
        print("lr of epoch", epoch, "=>", scheduler.get_last_lr())
        print('epoch: {} loss: {:.5f} y_now: {}'.format(epoch, loss.item(), y_now))

# predict
y_pre = model(torch.tensor(data[:, 0])).detach().numpy()

# save data
# np.save('Microphone_FNN_Full_States_Prediction.npy', y_pre)

print('Train_Y', RMSE_VAF(data[0:y_now, 1], y_pre[:y_now,0]))
print('Train_C', RMSE_VAF(data[0:y_now, 2], y_pre[:y_now,1]))

plt.figure(num=1, figsize=(16, 10))
plt.plot(np.arange(1, len(Fs_signal)+1),y_output,label='real y')
plt.plot(np.arange(1, len(Fs_signal)+1),c_output,label='real c')
plt.plot(np.arange(1, len(Fs_signal)+1),y_pre[:y_now,0],label='predicted y')
plt.plot(np.arange(1, len(Fs_signal)+1),y_pre[:y_now,1],label='predicted c')
plt.legend()
plt.title('train')

print('TEST_Y', RMSE_VAF(data[y_now:, 1], y_pre[y_now:,0]))
print('TEST_C', RMSE_VAF(data[y_now:, 2], y_pre[y_now:,1]))

plt.figure(num=2, figsize=(16, 10))
plt.plot(np.arange(len(Fs_signal)+1, 1001),data[y_now:,1],label='real y')
plt.plot(np.arange(len(Fs_signal)+1, 1001),data[y_now:,2],label='real c')
plt.plot(np.arange(len(Fs_signal)+1, 1001),y_pre[y_now:,0],label='predicted y')
plt.plot(np.arange(len(Fs_signal)+1, 1001),y_pre[y_now:,1],label='predicted c')
plt.legend()
plt.title('test')

plt.show()




