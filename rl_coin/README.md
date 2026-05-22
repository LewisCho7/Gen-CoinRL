강화학습 ppo코드입니다. 코랩에서 git clone해서 사용할 수 있을거에

Gen-CoinRL/envs/environment.yml 파일이 실행 환경 파일 입니다.
conda 가상환경 만들어서 사용하시면 됩니다. // 코랩이랑 gpu서버의 환경이 달라서 안돌아갈 수 있어요 그러면 환경 조정하시면 됩니다.

Gen-CoinRL/rl_coin/coinrun/train_agent.py가 학습 실행 코드 입니다.

아래 명령으로 실행 가능합니다.
python -m coinrun.train_agent --run-id 1000lv_128ne_nojump-action -nlev 1000 -ne 128
// --run-id인자는 저장할 모델 이름입니다.
// -nlev인자는 training에 사용할 맵 개수 입니다.
// -ne는 동시에 학습할 agent의 개수 입니다.


Gen-CoinRL/rl_coin/coinrun/PPO2.py가 실제 ppo 알고리즘 작성된 코드 입니다.

Gen-CoinRL/rl_coin/coinrun/coinrun.cpp에서
- 40~49번째 줄까지 => action을 조정할 수 있습니다.
- 1202~1223 번째 줄까지 => 보상 함수를 조정할 수 있습니다.
