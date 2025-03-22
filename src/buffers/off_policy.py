from typing import Union
from collections import deque
import numpy as np
import torch

class OffPolicyBuffer:
    def __init__(self, size: int, state_z_dim: int, state_vec_dim: int, action_dim: int, n_step: int = 3, gamma: float = 0.99):
        self.size = size
        self.state_z_dim = state_z_dim
        self.state_vec_dim = state_vec_dim
        self.action_dim = action_dim
        self.n_step = n_step
        self.gamma = gamma
        self.temp_buffer = deque(maxlen=n_step)
        
        self.buffer = {
            "state_z": np.zeros((size, state_z_dim), dtype=np.float32),
            "state_vec": np.zeros((size, state_vec_dim), dtype=np.float32),
            "action": np.zeros((size, action_dim), dtype=np.float32),
            "reward": np.zeros((size, 1), dtype=np.float32),
            "next_state_z": np.zeros((size, state_z_dim), dtype=np.float32),
            "next_state_vec": np.zeros((size, state_vec_dim), dtype=np.float32),
            "done": np.zeros((size, 1), dtype=np.bool_)
        }
        
        self.position = 0
        self.full = False

    def add(self, state_z: torch.Tensor, state_vec: torch.Tensor, action: torch.Tensor, reward: float, 
            next_state_z: torch.Tensor, next_state_vec: torch.Tensor, done: bool):
        self.temp_buffer.append((state_z, state_vec, action, reward, next_state_z, next_state_vec, done))
        if len(self.temp_buffer) >= self.n_step:
            self._store_n_step_transition()
        if done:
            while self.temp_buffer:
                self._store_n_step_transition()

    def _store_n_step_transition(self):
        state_z, state_vec, action, _, _, _, _ = self.temp_buffer[0]
        total_reward = 0.0
        discount = 1.0
        next_state_z, next_state_vec, done_flag = None, None, False

        for i, (_, _, _, r, ns_z, ns_vec, d) in enumerate(self.temp_buffer):
            total_reward += discount * r
            discount *= self.gamma
            next_state_z = ns_z
            next_state_vec = ns_vec
            done_flag = d
            if d:
                break  # 中断している場合はここで終わる

        idx = self.position
        self.buffer["state_z"][idx] = self._to_numpy(state_z)
        self.buffer["state_vec"][idx] = self._to_numpy(state_vec)
        self.buffer["action"][idx] = self._to_numpy(action)
        self.buffer["reward"][idx] = np.float32(total_reward)
        self.buffer["next_state_z"][idx] = self._to_numpy(next_state_z)
        self.buffer["next_state_vec"][idx] = self._to_numpy(next_state_vec)
        self.buffer["done"][idx] = done_flag

        self.position = (self.position + 1) % self.size
        if self.position == 0:
            self.full = True

        self.temp_buffer.popleft()

    def sample(self, batch_size: int, as_tensor: bool = True, device: str = "cpu") -> dict:
        max_idx = self.size if self.full else self.position
        indices = np.random.choice(max_idx, batch_size, replace=False)
        batch = {key: self.buffer[key][indices] for key in self.buffer}
        if as_tensor:
            batch = {k: torch.tensor(v, dtype=torch.float32).to(device) for k, v in batch.items()}
            batch["done"] = batch["done"].bool()  # doneだけbool型で明示
        return batch

    def __len__(self) -> int:
        return self.size if self.full else self.position

    def clear(self):
        self.position = 0
        self.full = False
        self.temp_buffer.clear()
        for key in self.buffer:
            self.buffer[key][:] = 0

    def _to_numpy(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return data
