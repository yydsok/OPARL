import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from sympy.abc import epsilon
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def tensormin(targetQ):
    Tensormin = targetQ[0]
    for i in targetQ:
         if torch.lt(Tensormin,i)[0] is False:
                Tensormin = i
    return Tensormin

def tensormax(targetQ):
    Tensormax = targetQ[0]
    for i in targetQ:
         if torch.gt(Tensormax,i)[0] is False:
                Tensormax = i
    return Tensormax


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class EnsembleLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, ensemble_size=7):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.register_parameter('weight', torch.nn.Parameter(torch.zeros(ensemble_size, in_features, out_features)))
        self.register_parameter('bias', torch.nn.Parameter(torch.zeros(ensemble_size, 1, out_features)))
        self.register_parameter('bias_1', torch.nn.Parameter(torch.zeros(ensemble_size, out_features)))
        torch.nn.init.trunc_normal_(self.weight, std=1/(2*in_features**0.5))
        self.register_parameter('saved_weight', torch.nn.Parameter(self.weight.detach().clone()))
        self.register_parameter('saved_bias', torch.nn.Parameter(self.bias.detach().clone()))
        self.select = list(range(0, self.ensemble_size))

    def forward(self, x):
        weight = self.weight[self.select]
        bias = self.bias[self.select]
        bias_1 = self.bias_1[self.select]
        if len(x.shape) == 2:
            x = torch.einsum('bi,eio->ebo', x, weight)
            return x + bias
        else:
            x = torch.einsum('ebi,eio->ebo', x, weight)
            return x + bias

    def set_select(self, indexes):
        assert len(indexes) <= self.ensemble_size and max(indexes) < self.ensemble_size
        self.select = indexes
        self.weight.data[indexes] = self.saved_weight.data[indexes]
        self.bias.data[indexes] = self.saved_bias.data[indexes]

    def update_save(self, indexes):
        self.saved_weight.data[indexes] = self.weight.data[indexes]
        self.saved_bias.data[indexes] = self.bias.data[indexes]


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class EnsembleCritic(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, ensemble_size=7, hidden_features=256, hidden_layers=2):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.ensemble_size = ensemble_size
        self.select = list(range(ensemble_size))

        self.activation = Swish()

        module_list = []
        for i in range(hidden_layers):
            if i == 0:
                module_list.append(EnsembleLinear(obs_dim + action_dim, hidden_features, ensemble_size))
            else:
                module_list.append(EnsembleLinear(hidden_features, hidden_features, ensemble_size))
        self.backbones = torch.nn.ModuleList(module_list)

        self.output_layer = EnsembleLinear(hidden_features, 1, ensemble_size)

    def forward(self, state, action):
        output = torch.cat([state, action], dim=-1)
        for layer in self.backbones:
            output = self.activation(layer(output))
        qvalue = self.output_layer(output)
        return qvalue

    def set_select(self, indexes):
        self.select = indexes
        for layer in self.backbones:
            layer.set_select(indexes)
        self.output_layer.set_select(indexes)

    def update_save(self, indexes):
        for layer in self.backbones:
            layer.update_save(indexes)
        self.output_layer.update_save(indexes)

# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim, Q_num):
#         super(Critic, self).__init__()
#         self.Q_num = Q_num
#         self.Ql1 = nn.ModuleList([nn.Linear(state_dim + action_dim, 256) for i in range(Q_num)])
#         self.Ql2 = nn.ModuleList([nn.Linear(256, 256) for i in range(Q_num)])
#         self.Ql3 = nn.ModuleList([nn.Linear(256, 1) for i in range(Q_num)])
#
#         # Q1 architecture
#         self.l1 = nn.Linear(state_dim + action_dim, 256)
#         self.l2 = nn.Linear(256, 256)
#         self.l3 = nn.Linear(256, 1)
#
#         # Q2 architecture
#         self.l4 = nn.Linear(state_dim + action_dim, 256)
#         self.l5 = nn.Linear(256, 256)
#         self.l6 = nn.Linear(256, 1)
#
#     def all_forward(self,state,action):
#         q = []
#         for i in range(self.Q_num):
#             sa = torch.cat([state, action], 1)
#             q1 = F.relu(self.Ql1[i](sa))
#             q1 = F.relu(self.Ql2[i](q1))
#             q1 = self.Ql3[i](q1)
#             q.append(q1)
#         return q
#
#     def multi_forward(self,state,action,Q_id):
#         sa = torch.cat([state, action], 1)
#         q1 = F.relu(self.Ql1[Q_id](sa))
#         q1 = F.relu(self.Ql2[Q_id](q1))
#         q1 = self.Ql3[Q_id](q1)
#         return q1


class TD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            reset_freq=2000,
            reset_exit_freq=2000,
            Q_num=5,
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.exit_actor = Actor(state_dim, action_dim, max_action).to(device)
        self.exit_actor_target = copy.deepcopy(self.exit_actor)
        self.exit_actor_optimizer = torch.optim.Adam(self.exit_actor.parameters(), lr=3e-4)

        self.critic = EnsembleCritic(state_dim, action_dim, Q_num).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.Q_num = Q_num
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.reset_freq = reset_freq
        self.reset_exit_freq = reset_exit_freq
        self.total_it = 0

    def select_action1(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def select_action2(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.exit_actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q, dim=0)[0]
            target_Q = reward + not_done * self.discount * target_Q
            
        current_Q = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q, target_Q.unsqueeze(0).repeat(self.Q_num, 1, 1))

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.reset_freq == 0:
            self.exit_actor.load_state_dict(self.actor.state_dict())

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_action = self.actor(state)
     
            Qmin = self.critic(state, actor_action)
            Qmin = torch.min(Qmin, dim=0)[0]
            actor_loss = - Qmin.mean()

            exit_action = self.exit_actor(state)
            Qmax = self.critic(state, exit_action)
            Qmax = torch.max(Qmax, dim=0)[0]
            exit_actor_loss = -Qmax.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.exit_actor_optimizer.zero_grad()
            exit_actor_loss.backward()
            self.exit_actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            # for param, target_param in zip(self.exit_actor.parameters(), self.exit_actor_target.parameters()):
            #     target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


            return {
                "critic_loss": critic_loss.item(),
                "actor_loss": actor_loss.item(),
                "exit_actor_loss": exit_actor_loss.item(),
            }

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

        torch.save(self.exit_actor.state_dict(), filename + "exit_actor")
        torch.save(self.exit_actor_optimizer.state_dict(), filename + "exit_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

        self.exit_actor.load_state_dict(torch.load(filename + "exit_actor"))
        self.exit_actor_optimizer.load_state_dict(torch.load(filename + "exit_actor_optimizer"))
        self.exit_actor_target = copy.deepcopy(self.exit_actor)



