import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TD3Agent:
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        actor_target: nn.Module,
        critic_target: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        n_step: int = 3,
        device: str = "cpu"
    ):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.actor_target = actor_target.to(device)
        self.critic_target = critic_target.to(device)
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.n_step = n_step
        # n-stepの割引項は、バッファ側で既に報酬に対してsum_{k=0}^{n-1} gamma^k r_{t+k}となっているので、
        # 次状態に対する割引は gamma^n とします。
        self.gamma_n = gamma ** n_step

        self.total_it = 0
        self.device = device

        # ターゲットネットワークを初期化
        self._hard_update(self.actor_target, self.actor)
        self._hard_update(self.critic_target, self.critic)

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """現在の状態（連結済みのstate_zとstate_vec）に対する行動を決定"""
        with torch.no_grad():
            action = self.actor(state.to(self.device))
        return action.cpu().numpy()

    def update(self, replay_buffer, batch_size: int = 64):
        self.total_it += 1

        # バッチサンプル（state_z, state_vecはそれぞれの次元で取得）
        batch = replay_buffer.sample(batch_size, as_tensor=True, device=self.device)
        # 状態はstate_zとstate_vecを連結して使用
        state = torch.cat([batch["state_z"], batch["state_vec"]], dim=1)
        next_state = torch.cat([batch["next_state_z"], batch["next_state_vec"]], dim=1)
        action = batch["action"]
        reward = batch["reward"]
        done = batch["done"].float()  # bool -> float (0.0 or 1.0)

        with torch.no_grad():
            # ターゲットアクションにノイズを加える
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)
            
            # ターゲットQ値の計算（2つのQ値のうち小さいほうを採用）
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            
            # n-step報酬が既にバッファに保存されているので、次状態に対する割引はgamma^nを乗じる
            target_Q = reward + (self.gamma_n * (1 - done)) * target_Q

        # 現在のQ値を計算
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Policy updateは一定の遅延後に行う
        if self.total_it % self.policy_delay == 0:
            # Actorの損失は、criticが出力するQ値（Q1）を最大化する方向
            actor_loss = -self.critic(state, self.actor(state))[0].mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ターゲットネットワークのソフト更新
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

    def _soft_update(self, net: nn.Module, target_net: nn.Module):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def _hard_update(self, target_net: nn.Module, net: nn.Module):
        target_net.load_state_dict(net.state_dict())
