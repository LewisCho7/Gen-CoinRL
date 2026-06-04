import os
import sys
import random
import time
import csv
import collections # 파이썬 기본 모듈 (ReplayBuffer용)
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 🌟 CoinRun 환경 라이브러리 추가
sys.path.insert(0, os.path.abspath('.'))
from coinrun import setup_utils
from coinrun.coinrunenv import make as make_env


# -----------------------------------------------------------
# 🌟 CleanRL 의존성을 없애기 위해 직접 구현한 완벽 독립형 ReplayBuffer
class ReplayBuffer:
    def __init__(self, capacity, obs_space, action_space, device, handle_timeout_termination=False):
        self.buffer = collections.deque(maxlen=capacity)
        self.device = device

    def add(self, obs, next_obs, action, reward, done, info):
        # 코인런은 num_envs=1 이므로 [0]번째 데이터를 추출해 저장합니다.
        self.buffer.append((obs[0], next_obs[0], action[0], reward[0], done[0]))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, next_obs, action, reward, done = zip(*batch)
        
        # 학습 루프에서 데이터에 쉽게 접근(data.observations 등)하도록 돕는 미니 데이터 클래스
        class BatchData:
            def __init__(self, o, no, a, r, d, device):
                self.observations = torch.tensor(np.array(o), dtype=torch.float32).to(device)
                self.next_observations = torch.tensor(np.array(no), dtype=torch.float32).to(device)
                self.actions = torch.tensor(np.array(a), dtype=torch.int64).unsqueeze(1).to(device)
                self.rewards = torch.tensor(np.array(r), dtype=torch.float32).to(device)
                self.dones = torch.tensor(np.array(d), dtype=torch.float32).to(device)
                
        return BatchData(obs, next_obs, action, reward, done, self.device)
# -----------------------------------------------------------


@dataclass
class Args:
    exp_name: str = "cleanrl_dqn"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    save_model: bool = True 

    # Algorithm specific arguments
    env_id: str = "standard"
    total_timesteps: int = 1000000 
    learning_rate: float = 2.5e-4
    num_envs: int = 1
    
    # 🌟 100만 스텝 맞춤형 황금 밸런스 수정 부분
    buffer_size: int = 100000 
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 10000
    batch_size: int = 32
    start_e: float = 1
    end_e: float = 0.05
    exploration_fraction: float = 0.5
    learning_starts: int = 10000
    train_frequency: int = 4
    
    # 🌟 추가된 부분: Validation(검증) 세팅
    eval_frequency: int = 20000  # 2만 스텝마다 모의고사를 실시
    eval_episodes: int = 10      # 모의고사는 10판을 진행하여 평균을 냄
    num_levels: int = 500
    set_seed: int = 0


# -----------------------------------------------------------
# 🌟 파이토치로 완벽 이식된 순정 IMPALA CNN (Batch Norm, Dropout 제거)
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # TF의 padding='same', kernel=3 은 PyTorch의 padding=1 과 동일합니다.
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # TF 원본: out = tf.nn.relu(inputs) -> conv_layer -> relu -> conv_layer
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        return out + x  # Residual connection

class ConvSequence(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # TF 원본: max_pooling2d (pool_size=3, strides=2, padding='same')
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

class QNetwork(nn.Module):
    def __init__(self, action_space_n):
        super().__init__()
        # 1. IMPALA CNN 구조 (깊이: 16 -> 32 -> 32)
        self.seq1 = ConvSequence(3, 16)
        self.seq2 = ConvSequence(16, 32)
        self.seq3 = ConvSequence(32, 32)
        
        # 2. Flatten 이후의 Dense 레이어 (64x64 이미지가 8x8로 축소됨)
        self.fc = nn.Linear(32 * 8 * 8, 256)
        
        # 3. DQN 행동 가치(Q-value) 출력 레이어
        self.q_head = nn.Linear(256, action_space_n)

    def forward(self, x):
        # 파이토치에 맞게 차원 변경 (NHWC -> NCHW) 및 0~1 정규화
        x = x.permute(0, 3, 1, 2).float() / 255.0
        
        x = self.seq1(x)
        x = self.seq2(x)
        x = self.seq3(x)
        
        x = torch.flatten(x, 1)
        x = F.relu(x)
        x = F.relu(self.fc(x))
        
        return self.q_head(x)
# -----------------------------------------------------------

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = Args()
    run_name = f"CoinRun__{args.env_id}__{args.exp_name}__{args.seed}"

    # 세팅 및 시드 고정
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"🔥 사용 중인 디바이스: {device}")

    # 코인런 환경 생성
    os.environ['COINRUN_RESOURCES_PATH'] = os.path.abspath('./coinrun/assets')
    setup_utils.setup_and_load(
        num_envs=args.num_envs, 
        is_high_res=False, 
        game_type=args.env_id,
        num_levels=args.num_levels, 
        set_seed=args.set_seed
    )
    envs = make_env(args.env_id, num_envs=args.num_envs)

    q_network = QNetwork(envs.action_space.n).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs.action_space.n).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.observation_space, 
        envs.action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    ep_rewards = np.zeros(args.num_envs)
    ep_lengths = np.zeros(args.num_envs)

    # 🌟 CSV 로깅 세팅
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    
    rewards_csv = os.path.join(log_dir, f"{run_name}_rewards.csv")
    losses_csv = os.path.join(log_dir, f"{run_name}_losses.csv")
    val_csv = os.path.join(log_dir, f"{run_name}_validation.csv") # 🌟 검증용 CSV 추가

    with open(rewards_csv, mode='w', newline='') as f:
        csv.writer(f).writerow(["global_step", "episode_reward", "episode_length", "epsilon"])
    with open(losses_csv, mode='w', newline='') as f:
        csv.writer(f).writerow(["global_step", "q_loss", "avg_q_value", "SPS"])
    with open(val_csv, mode='w', newline='') as f:
        csv.writer(f).writerow(["global_step", "avg_val_reward", "avg_val_length"]) # 🌟 검증 헤더 추가

    print("🚀 학습을 시작합니다!")
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        
        # ---------------------------------------------------------
        # 🌟 주기적 Validation(검증) 로직 (새로 추가된 부분)
        # ---------------------------------------------------------
        if global_step > 0 and global_step % args.eval_frequency == 0:
            print(f"\n🕵️‍♂️ [Step {global_step}] 검증(Validation) 시작 (총 {args.eval_episodes}판)...")
            q_network.eval() # 뇌를 평가 모드로 전환 (안전장치)
            
            val_rewards = []
            val_lengths = []
            val_ep_ret = 0.0
            val_ep_len = 0
            
            val_obs = envs.reset() # 검증을 위해 맵 초기화
            while len(val_rewards) < args.eval_episodes:
                # 검증 시에는 Epsilon을 최소화(0.05)하여 진짜 실력을 측정
                if random.random() < args.end_e:
                    val_actions = np.array([envs.action_space.sample() for _ in range(args.num_envs)])
                else:
                    with torch.no_grad():
                        val_q = q_network(torch.Tensor(val_obs).to(device))
                        val_actions = torch.argmax(val_q, dim=1).cpu().numpy()
                
                val_obs, val_r, val_d, _ = envs.step(val_actions)
                
                val_ep_ret += val_r[0]
                val_ep_len += 1
                
                if val_d[0]:
                    val_rewards.append(val_ep_ret)
                    val_lengths.append(val_ep_len)
                    val_ep_ret = 0.0
                    val_ep_len = 0
                    
            avg_val_reward = np.mean(val_rewards)
            avg_val_length = np.mean(val_lengths)
            print(f"📊 [검증 결과] 평균 점수: {avg_val_reward:.2f}/10점 | 평균 생존: {avg_val_length:.1f} 프레임\n")
            
            # 검증 결과를 CSV에 저장
            with open(val_csv, mode='a', newline='') as f:
                csv.writer(f).writerow([global_step, avg_val_reward, avg_val_length])
            
            q_network.train() # 뇌를 다시 학습 모드로 전환
            
            # 🌟 매우 중요: 검증이 끝난 후 본 학습 루프를 오염시키지 않기 위해 환경 및 점수 초기화
            obs = envs.reset()
            ep_rewards.fill(0)
            ep_lengths.fill(0)
        # ---------------------------------------------------------

        # 본 학습 루프 진행
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        
        if random.random() < epsilon:
            actions = np.array([envs.action_space.sample() for _ in range(args.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_obs, rewards, dones, infos = envs.step(actions)
        
        # 에피소드 단위 처리 및 CSV 로깅
        for idx, d in enumerate(dones):
            ep_rewards[idx] += rewards[idx]
            ep_lengths[idx] += 1
            if d:
                with open(rewards_csv, mode='a', newline='') as f:
                    csv.writer(f).writerow([global_step, ep_rewards[idx], ep_lengths[idx], epsilon])
                ep_rewards[idx] = 0.0
                ep_lengths[idx] = 0

        rb.add(obs, next_obs, actions, rewards, dones, infos) 
        obs = next_obs

        # 학습(역전파) 로직
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    # 1. 다음 상태(s')에서 가장 높은 Q값 찾기
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    # 2. 정답지(Target) 만들기 (벨만 방정식) - 게임 오버시(dones=1) 미래 보상 차단
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                
                # 3. 과거에 내가 예측했던 Q값 가져오기
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                
                # 4. 오차(Loss) 계산 (Vanilla DQN MSE)
                loss = F.mse_loss(td_target, old_val)
                avg_q = old_val.mean().item()

                if global_step % 100 == 0:
                    sps = int(global_step / (time.time() - start_time))
                    print(f"Step: {global_step}/{args.total_timesteps} | Loss: {loss.item():.4f} | Avg Q: {avg_q:.4f} | SPS: {sps}")
                    
                    with open(losses_csv, mode='a', newline='') as f:
                        csv.writer(f).writerow([global_step, loss.item(), avg_q, sps])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 타겟 네트워크 업데이트
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    # 🌟 학습 완료 후 지정된 폴더에 모델 가중치 저장
    if args.save_model:
        model_dir = f"saved_models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{run_name}.pth")
        torch.save(q_network.state_dict(), model_path)
        print(f"✅ 학습 완료! 모델이 저장되었습니다: {model_path}")

    envs.close()