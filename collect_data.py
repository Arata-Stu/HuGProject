import os
import cv2
import numpy as np
import hydra
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import pygame
from omegaconf import DictConfig, OmegaConf

from src.envs.car_racing import CarRacingWithInfoWrapper
from src.utils.timers import Timer as Timer

@hydra.main(config_path="config", config_name="collect_data", version_base="1.2")
def main(config: DictConfig):
    print('------ Configuration ------')
    print(OmegaConf.to_yaml(config))
    print('---------------------------')

    # 環境のセットアップ
    env_cfg = config.env
    env = gym.make(env_cfg.name, render_mode=env_cfg.render_mode)
    env = TimeLimit(env, max_episode_steps=env_cfg.num_steps)
    env = CarRacingWithInfoWrapper(env=env, width=env_cfg.width, height=env_cfg.height)

    mode = config.mode  # 'manual' または 'random'

    # pygame の初期化（manual モードのみ）
    if mode == "manual":
        pygame.init()
        screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("CarRacing Control")
        clock = pygame.time.Clock()

    # 画像保存用のベースディレクトリ
    save_dir = config.output_dir
    os.makedirs(save_dir, exist_ok=True)

    max_episodes = config.num_episodes
    max_steps = config.num_steps

    episode_count = 0
    while episode_count < max_episodes:
        obs, info = env.reset()
        step = 0
        episode_reward = 0
        done = False

        # エピソードごとのディレクトリ作成
        episode_dir = os.path.join(save_dir, f"ep{episode_count:03d}")
        os.makedirs(episode_dir, exist_ok=True)

        while not done and step < max_steps:
            if mode == "manual":
                screen.fill((0, 0, 0))
                keys = pygame.key.get_pressed()
                action = np.array([0.0, 0.0, 0.0])

                if keys[pygame.K_LEFT]:
                    action[0] = -0.3
                if keys[pygame.K_RIGHT]:
                    action[0] = 0.3
                if keys[pygame.K_UP]:
                    action[1] = 0.3
                if keys[pygame.K_DOWN]:
                    action[2] = 0.3

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        env.close()
                        return
                clock.tick(30)
            
            else:  # 'random' モード
                action = env.action_space.sample()

            # 環境をステップ実行
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            if isinstance(obs, dict):
                obs = obs["image"]

            if isinstance(obs, np.ndarray):
                img_path = os.path.join(episode_dir, f"step{step:04d}.png")
                cv2.imwrite(img_path, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

            step += 1

        print(f"Episode {episode_count + 1}/{max_episodes} finished. Total reward: {episode_reward:.2f}")
        episode_count += 1

    env.close()
    if mode == "manual":
        pygame.quit()

if __name__ == "__main__":
    main()
