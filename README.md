# Gen-CoinRL
Creating and testing generalization of agent in Open-ai Procgen CoinRun environment using various deep reinforcement algorithms(DQN, PPO, SAC)

- src/ dqn, ppo, sac 우리 환경에 맞게 알고리즘 구현
- envs/wrappers.py -> 보상 함수 설계
- helpers.py -> WandB 초기 설정, 비디오 logging 저장 등

bash
₩python main.py --algo PPO --shaping False₩
