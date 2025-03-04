import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR


torch.manual_seed(1)

B = 1.0
m = 0.2
K = 1.0
e_A = 0.5
R = 1.0
L = 1.0

k1 = -B/m
k2 = -K/m
k3 = 1/e_A
k4 = 1/m
k5 = -R/L
k6 = 1/L


y0 = 1
u = torch.tensor(1.0)
h = 0.1
x0 = torch.tensor([0.0, 0.0, 0.0, 0.0])
epsilon = 0.002


class Potential_NN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Potential_NN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, q):

        return self.network(q)
        # return 1/(2*e_A)*(y0-q[0])*q[1]**2 + 1/2*K*q[0]**2


class Kinetic_Matrix_NN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Kinetic_Matrix_NN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, 2),
            nn.Softplus(),
        )
        # self.m = nn.Parameter(torch.tensor(5.0))
        # self.L = nn.Parameter(torch.tensor(1.0))
        self.m = torch.tensor(5.0)
        self.L = torch.tensor(1.0)


    def forward(self, q):
        dig_ele = self.network(q) + epsilon
        kinetic_matrix = torch.diag(dig_ele)

        return kinetic_matrix
        # return torch.diag(torch.cat((self.m.view(1), self.L.view(1))))


class Damping_Matrix_NN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Damping_Matrix_NN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, 2),
            nn.Softplus(),
        )

        # self.B = nn.Parameter(torch.tensor(1.0))
        # self.R = nn.Parameter(torch.tensor(1.0))
        self.B = torch.tensor(1.0)
        self.R = torch.tensor(1.0)

    def forward(self, q):
        dig_ele = self.network(q) + epsilon
        damping_matrix = torch.diag(dig_ele)

        return damping_matrix
        # return torch.diag(torch.cat((self.B.view(1), self.R.view(1))))


def dynamic_matrices(q):
    M_inv = Kinetic_matrix_model(q)
    D_mat = Damping_matrix_model(q)

    new_q = q.clone().detach().requires_grad_(True)
    u_energy = Potential_model(new_q)
    du_dq = torch.autograd.grad(u_energy, new_q, create_graph=True)[0]

    return M_inv, D_mat, du_dq


def system_fun(q, q_dot, Fs):
    M_inv, D_mat, du_dq = dynamic_matrices(q)
    q_dot_next = M_inv @ (torch.stack([Fs.to(torch.float32), u]) - D_mat@q_dot - du_dq*torch.tensor([1.0, -1.0]))

    return q_dot, q_dot_next


def RK_step(q, q_dot, Fs):
    k1_q, k1_q_dot = system_fun(q, q_dot, Fs)
    k2_q, k2_q_dot = system_fun(q + h * k1_q / 2, q_dot + h * k1_q_dot / 2, Fs)
    k3_q, k3_q_dot = system_fun(q + h * k2_q / 2, q_dot + h * k2_q_dot / 2, Fs)
    k4_q, k4_q_dot = system_fun(q + h * k3_q, q_dot + h * k3_q_dot, Fs)

    q_next = q + 1/6*(k1_q+2*k2_q+2*k3_q+k4_q)*h
    q_dot_next = q_dot + 1/6*(k1_q_dot+2*k2_q_dot+2*k3_q_dot+k4_q_dot)*h

    return q_next, q_dot_next


def Preditc(Fs):
    qpre = []
    qpre_dot = []

    q = torch.tensor([0.0, 0.0])
    q_dot = torch.tensor([0.0, 0.0])

    for i in range(len(Fs)):
        q_next, q_dot_next = RK_step(q, q_dot, Fs[i])
        qpre.append(q_next)
        qpre_dot.append(q_dot_next)
        q = q_next
        q_dot = q_dot_next

    return torch.stack(qpre), torch.stack(qpre_dot)


def RMSE_VAF(y, y_hat):
    RMSE = np.linalg.norm(y-y_hat)/np.sqrt(len(y))
    VAF = 1-np.linalg.norm(y-y_hat)**2/np.linalg.norm(y)**2
    return RMSE, VAF


def rule(epoch):
    if epoch < warm_up_step:
        return epoch/warm_up_step
    else:
        return pow(gamma, int((epoch-warm_up_step)/25))


LEARNING_RATE = 0.001
Epoch = 15000
warm_up_step = 5000
gamma = 0.99

y_num = 500
y_now = 10
# start from 100 data points
data = np.load('data/ChirpStimulation.npy')
Fs_signal = torch.tensor(data[0:y_now,0])
y_output = torch.tensor(data[0:y_now,1])
c_output = torch.tensor(data[0:y_now,2])


hidden_num = 32
Potential_model = Potential_NN(2, hidden_num)
Kinetic_matrix_model = Kinetic_Matrix_NN(2, hidden_num)
Damping_matrix_model = Damping_Matrix_NN(2, hidden_num)

optimizer = optim.Adam(list(Potential_model.parameters()) +
                       list(Kinetic_matrix_model.parameters()) + list(Damping_matrix_model.parameters()), lr=LEARNING_RATE)
scheduler = LambdaLR(optimizer, lr_lambda=rule)
tic = 0
for i in range(Epoch):

    if y_now > 50:
        tic += 1

    qpre, qpre_dot = Preditc(Fs_signal)
    loss = (qpre[:,0] - y_output).pow(2).sum()+(qpre[:,1] - c_output).pow(2).sum()

    if loss <= 15:
        if y_now < y_num:
            tic = 0
            y_now += 10
            Fs_signal = torch.tensor(data[0:y_now, 0])
            y_output = torch.tensor(data[0:y_now, 1])
            c_output = torch.tensor(data[0:y_now, 2])

    if tic == 250:
        if y_now < y_num:
            tic = 0
            y_now += 10
            Fs_signal = torch.tensor(data[0:y_now, 0])
            y_output = torch.tensor(data[0:y_now, 1])
            c_output = torch.tensor(data[0:y_now, 2])
        else:
            break


    if i % 10 == 0:
        print("lr of epoch", i, "=>", scheduler.get_last_lr())
        print('epoch: {} loss: {:.5f} y_now: {}'.format(i, loss.item(), y_now))  # loss为(1,)的tensor

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(list(Potential_model.parameters()) +
                       list(Kinetic_matrix_model.parameters()) + list(Damping_matrix_model.parameters()), max_norm=1.0)

    optimizer.step()
    scheduler.step()

# drawing the result
qpre, qpre_dot = Preditc(torch.tensor(data[:,0]))
qpre = qpre.detach().numpy()
qpre_dot = qpre_dot.detach().numpy()

# save data
np.save('Simulation_DeLaN_Full_States_Prediction.npy', qpre)
qpre = np.load("Simulation_DeLaN_Full_States_Prediction.npy")
print('Train_Y', RMSE_VAF(data[0:y_now, 1], qpre[:y_now,0]))
print('Train_C', RMSE_VAF(data[0:y_now, 2], qpre[:y_now,1]))

plt.figure(num=1, figsize=(16,10))
plt.plot(np.arange(1, len(Fs_signal)+1),y_output,label='real y')
plt.plot(np.arange(1, len(Fs_signal)+1),c_output,label='real c')
plt.plot(np.arange(1, len(Fs_signal)+1),qpre[:y_now,0],label='predicted y')
plt.plot(np.arange(1, len(Fs_signal)+1),qpre[:y_now,1],label='predicted c')
plt.legend()
plt.title('train')

print('TEST_Y', RMSE_VAF(data[y_now:, 1], qpre[y_now:,0]))
print('TEST_C', RMSE_VAF(data[y_now:, 2], qpre[y_now:,1]))

plt.figure(num=2, figsize=(16,10))
plt.plot(np.arange(len(Fs_signal)+1, 1001),data[y_now:,1],label='real y')
plt.plot(np.arange(len(Fs_signal)+1, 1001),data[y_now:,2],label='real c')
plt.plot(np.arange(len(Fs_signal)+1, 1001),qpre[y_now:,0],label='predicted y')
plt.plot(np.arange(len(Fs_signal)+1, 1001),qpre[y_now:,1],label='predicted c')
plt.legend()
plt.title('test')

plt.show()





