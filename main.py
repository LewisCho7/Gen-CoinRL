# main.py
import argparse
from procgen import ProcgenEnv
import wandb

# 💡 [핵심] 우리가 만든 파일에서 스태프들 불러오기!
from envs.wrappers import PBRSCoinRunWrapper
from src.ppo import PPOAgent
# from src.sac import SACAgent # SAC 실험할 때 주석 해제

def main():
    # 1. 실행할 때 어떤 알고리즘을 쓸지 명령어 인자(Argument)로 받기
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "SAC"])
    parser.add_argument("--shaping", type=bool, default=False)
    args = parser.parse_args()

    # 2. WandB 초기화 (팀 공간 연동)
    wandb.init(
        entity="your_team_name",
        project="coinrun-shaping",
        name=f"{args.algo}-Shaping_{args.shaping}"
    )

    # 3. CoinRun 기본 환경 생성
    base_env = ProcgenEnv(num_envs=1, env_name="coinrun", num_levels=500, start_level=0)
    
    # 4. 조건에 따라 우리가 만든 Reward Shaping Wrapper 씌우기
    if args.shaping:
        env = PBRSCoinRunWrapper(base_env)
    else:
        env = base_env

    # 5. 선택한 알고리즘에 맞는 두뇌 배정하기
    if args.algo == "PPO":
        agent = PPOAgent(num_actions=15)
    elif args.algo == "SAC":
        pass # agent = SACAgent(num_actions=15)

    # 6. 학습 루프 시작
    print(f"🚀 {args.algo} 학습을 시작합니다. (Reward Shaping: {args.shaping})")
    obs = env.reset()
    for step in range(1000000):
        # 에이전트가 행동 선택 -> 환경 step 진행 -> 학습 -> WandB 로그 기록
        # (여기에 메인 학습 루프 코드가 들어갑니다.)
        pass

if __name__ == "__main__":
    main()
