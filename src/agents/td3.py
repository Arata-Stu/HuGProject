import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.actor import ActorTD3
from src.models.critic import Critic

# TD3エージェントの定義（内部でモデルを生成）
class TD3Agent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        n_step: int = 3,
        device: str = "cpu",
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3
    ):
        self.device = device
        # ネットワークの生成
        self.actor = ActorTD3(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target = ActorTD3(state_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)

        # オプティマイザの生成
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.n_step = n_step
        # n-stepの場合、報酬はバッファ内でsum_{k=0}^{n-1} gamma^k * r_{t+k}となっているため、
        # 次状態に対する割引は gamma^n を用います
        self.gamma_n = gamma ** n_step

        self.total_it = 0

        # ターゲットネットワークの初期化（パラメータのハードコピー）
        self._hard_update(self.actor_target, self.actor)
        self._hard_update(self.critic_target, self.critic)

    def select_action(self, state: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            action = self.actor(state.to(self.device))
        return action.cpu().numpy()

    def update(self, replay_buffer, batch_size: int = 64):
        self.total_it += 1

        # バッチデータの取得（state_z, state_vec などはそれぞれの次元で取得）
        batch = replay_buffer.sample(batch_size, as_tensor=True, device=self.device)
        # 状態は必要に応じて連結して使用（例：state_zとstate_vec）
        state = torch.cat([batch["state_z"], batch["state_vec"]], dim=1)
        next_state = torch.cat([batch["next_state_z"], batch["next_state_vec"]], dim=1)
        action = batch["action"]
        reward = batch["reward"]
        done = batch["done"].float()  # bool → float

        with torch.no_grad():
            # ターゲットアクションにノイズを付加
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)
            
            # ターゲットQ値の計算（2つのQ値のうち小さい方を採用）
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (self.gamma_n * (1 - done)) * target_Q

        # 現在のQ値を計算
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor_loss の初期値を設定
        actor_loss_value = 0.0

        # 一定ステップ毎にActorの更新を実施
        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic(state, self.actor(state))[0].mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # ターゲットネットワークのソフトアップデート
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
            
            actor_loss_value = actor_loss.item()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss_value,
        }


    def _soft_update(self, net: nn.Module, target_net: nn.Module):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def _hard_update(self, target_net: nn.Module, net: nn.Module):
        target_net.load_state_dict(net.state_dict())
        
    # モデルの重みとオプティマイザの状態を保存するメソッド
    def save(self, filename: str):
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "total_it": self.total_it,
        }
        torch.save(checkpoint, filename)
    
    # 保存した重みとオプティマイザの状態をロードするメソッド
    def load(self, filename: str):
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.total_it = checkpoint.get("total_it", 0)
