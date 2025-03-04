import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from torch.optim.lr_scheduler import LambdaLR


y0 = 1
u = 1
h = 0.1
x0 = torch.tensor([0.0, 0.0, 0.0, 0.0])


def fun_full(x, Fs, k1, k2, k3, k4, k5, k6):  # State-space model for acoustic sensor
    return torch.stack([x[2],
                       x[3],
                       k1*x[2] + k2*x[0] + 0.5*k3*k4*(x[1]**2) + k4*Fs,
                       k5*x[3] - (x[0]-y0)*k3*k6*x[1] + k6*u])


def RK4_full(x, Fs, k1, k2, k3, k4, k5, k6):  # RK4
    K1 = fun_full(x, Fs, k1, k2, k3, k4, k5, k6)
    K2 = fun_full(x+h/2*K1, Fs, k1, k2, k3, k4, k5, k6)
    K3 = fun_full(x+h/2*K2, Fs, k1, k2, k3, k4, k5, k6)
    K4 = fun_full(x+h*K3, Fs, k1, k2, k3, k4, k5, k6)
    return x+h/6*(K1+2*K2+2*K3+K4)


class ID(nn.Module):
    def __init__(self, k1, k2, k3, k4, k5, k6):
        super(ID, self).__init__()
        self.k1 = nn.Parameter(torch.tensor(k1))
        self.k2 = nn.Parameter(torch.tensor(k2))
        self.k3 = nn.Parameter(torch.tensor(k3))
        self.k4 = nn.Parameter(torch.tensor(k4))
        self.k5 = nn.Parameter(torch.tensor(k5))
        self.k6 = nn.Parameter(torch.tensor(k6))

    def forward(self, u):
        ypre = []
        x = x0

        for i in range(len(u)):
            x_next = RK4_full(x, u[i], self.k1, self.k2, self.k3, self.k4, self.k5, self.k6)
            ypre.append(x_next)
            x = x_next

        return torch.stack(ypre)


def rule(epoch):
    if epoch < warm_up_step:
        return epoch/warm_up_step
    else:
        return pow(gamma, int((epoch-warm_up_step)/25))


y_num = 500
y_now = 100
# start from 100 data points
data = np.load('data/ChirpStimulation.npy')
Fs_signal = torch.tensor(data[0:y_now, 0])
y_output = torch.tensor(data[0:y_now, 1])
c_output = torch.tensor(data[0:y_now, 2])
var_y = np.var(y_output.numpy())
var_c = np.var(c_output.numpy())
coef_y = (var_y+var_c)/var_y
coef_c = (var_y+var_c)/var_c

LEARNING_RATE = 0.01
Epoch = 6400
warm_up_step = 800
gamma = 0.98

# id_sys = ID(-5.0, -5.0, 2.0, 5.0, -1.0, 1.0)
id_sys = ID(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

optimizer = optim.Adam(id_sys.parameters(), lr=LEARNING_RATE)
scheduler = LambdaLR(optimizer, lr_lambda=rule)

for i in range(Epoch):
    y_pre = id_sys(Fs_signal)
    loss = (y_pre[:, 0] - y_output).pow(2).sum()*coef_y + \
           (y_pre[:, 1] - c_output).pow(2).sum()*coef_c

    if loss <= 5:
        if y_now < y_num:
            y_now += 10
            Fs_signal = torch.tensor(data[0:y_now, 0])
            y_output = torch.tensor(data[0:y_now, 1])
            c_output = torch.tensor(data[0:y_now, 2])

    if i % 100 == 0:
        print("lr of epoch", i, "=>", scheduler.get_last_lr())
        print('epoch: {} loss: {:.5f} y_now: {}'.format(i, loss.item(), y_now))  # loss为(1,)的tensor

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(id_sys.parameters(), max_norm=1.0)

    optimizer.step()
    scheduler.step()

# (k1, k2, k3, k4, k5, k6)
print('k1 : ', id_sys.k1)
print('k2 : ', id_sys.k2)
print('k3 : ', id_sys.k3)
print('k4 : ', id_sys.k4)
print('k5 : ', id_sys.k5)
print('k6 : ', id_sys.k6)

y_pre = id_sys(torch.tensor(data[:,0])).detach().numpy()

# save data
# np.save('Simulation_RKNN_Full_States_Prediction.npy', y_pre)
print('-----------Training----------------')
y_output = y_output.detach().numpy()
c_output = c_output.detach().numpy()
print('RMSE-y:', np.linalg.norm(y_output-y_pre[:y_now,0])/np.sqrt(y_now))
print('RMSE-c:', np.linalg.norm(c_output-y_pre[:y_now,1])/np.sqrt(y_now))
print('VAF-y:', 1-np.linalg.norm(y_output-y_pre[:y_now,0])**2/np.linalg.norm(y_output)**2)
print('VAF-c:', 1-np.linalg.norm(c_output-y_pre[:y_now,1])**2/np.linalg.norm(c_output)**2)
print('-------------Testing--------------')

print('RMSE-y:', np.linalg.norm(data[y_now:,1]-y_pre[y_now:,0])/np.sqrt(1000-y_now))
print('RMSE-c:', np.linalg.norm(data[y_now:,2]-y_pre[y_now:,1])/np.sqrt(1000-y_now))
print('VAF-y:', 1-np.linalg.norm(data[y_now:,1]-y_pre[y_now:,0])**2/np.linalg.norm(data[y_now:,1])**2)
print('VAF-c:', 1-np.linalg.norm(data[y_now:,2]-y_pre[y_now:,1])**2/np.linalg.norm(data[y_now:,2])**2)

print('---------------------------')
plt.figure(num=1, figsize=(16,10))
plt.plot(np.arange(1, len(Fs_signal)+1),y_output,label='real y')
plt.plot(np.arange(1, len(Fs_signal)+1),c_output,label='real c')
plt.plot(np.arange(1, len(Fs_signal)+1),y_pre[:y_now,0],label='predicted y')
plt.plot(np.arange(1, len(Fs_signal)+1),y_pre[:y_now,1],label='predicted c')
plt.legend()
plt.title('train')

plt.figure(num=2, figsize=(16,10))
plt.plot(np.arange(len(Fs_signal)+1, 1001),data[y_now:,1],label='real y')
plt.plot(np.arange(len(Fs_signal)+1, 1001),data[y_now:,2],label='real c')
plt.plot(np.arange(len(Fs_signal)+1, 1001),y_pre[y_now:,0],label='predicted y')
plt.plot(np.arange(len(Fs_signal)+1, 1001),y_pre[y_now:,1],label='predicted c')
plt.legend()
plt.title('test')

plt.show()

L = 1/id_sys.k6
m = 1/id_sys.k4
print('B : ', -m*id_sys.k1)
print('m : ', m)
print('K : ', -m*id_sys.k2)
print('e_A : ', 1/id_sys.k3)
print('R : ', -id_sys.k5*L)
print('L : ', 1/id_sys.k6)
