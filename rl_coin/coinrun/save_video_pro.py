import os
import gym
import procgen
import numpy as np
import imageio
import tensorflow as tf
import joblib
from pyvirtualdisplay import Display

# 1. 가짜 객체 정의 (에러 방지용)
class DummyWriter:
    def __init__(self, *args, **kwargs): pass
    def log_scalar(self, *args, **kwargs): pass
    def add_summary(self, *args, **kwargs): pass

from coinrun import ppo2, config, tb_utils, policies
from coinrun.config import Config

ppo2.TB_Writer = DummyWriter
tb_utils.TB_Writer = DummyWriter

# 2. 디스플레이 및 세션 설정
display = Display(visible=0, size=(400, 300))
display.start()

config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config_tf)

try:
    # 3. 모델 설정
    Config.ARCHITECTURE = 'nature'
    Config.L2_WEIGHT = 1e-4
    Config.TEST = False
    Config.SYNC_FROM_ROOT = False
    
    # 4. 환경 생성
    print("Creating environment...")
    env_raw = gym.make("procgen:procgen-coinrun-v0", render_mode="rgb_array")
    
    # 5. 모델 구조 생성
    print("Building model structure...")
    model = ppo2.Model(
        policy=policies.CnnPolicy,
        ob_space=env_raw.observation_space,
        ac_space=env_raw.action_space,
        nbatch_act=1,
        nbatch_train=64,
        nsteps=256,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5
    )

    # 6. [가중치 로딩 로직 혁신] 이름 기반 매칭
    load_path = '/root/doyun_cap/ai/rl/coinrun/coinrun/saved_models/sav_1000lv_128ne_nojump_action_0'
    print(f"Loading weights from: {load_path}")
    
    data = joblib.load(load_path)
    raw_params = data['params'] 

    variables = tf.trainable_variables()
    restores = []

    if isinstance(raw_params, dict):
        print("Dictionary detected. Matching by variable names...")
        for v in variables:
            # 텐서플로우 변수 이름에서 ':0'을 떼고 파일 내 키와 매칭 시도
            clean_name = v.name.split(':')[0]
            # 'model/c1/w' 혹은 'c1/w' 형태의 키가 있는지 확인
            found_key = None
            for k in raw_params.keys():
                if k.endswith(clean_name) or clean_name.endswith(k):
                    found_key = k
                    break
            
            if found_key:
                loaded_p = np.array(raw_params[found_key])
                if v.shape.as_list() == list(loaded_p.shape):
                    restores.append(v.assign(loaded_p))
                else:
                    print(f"Shape mismatch for {v.name}: {v.shape} vs {loaded_p.shape}")
            else:
                print(f"Warning: Could not find weights for {v.name} in file.")
    else:
        # 리스트인 경우 순서대로 매칭
        for v, lp in zip(variables, raw_params):
            lp_np = np.array(lp)
            if v.shape.as_list() == list(lp_np.shape):
                restores.append(v.assign(lp_np))

    if restores:
        sess.run(restores)
        print(f"Successfully loaded {len(restores)} weight tensors! AI is fully awake.")
    else:
        print("Error: No weights were loaded. Check your model and file.")

    # 7. 플레이 및 비디오 프레임 추출
    obs = env_raw.reset()
    if len(obs.shape) == 3:
        obs = np.expand_dims(obs, axis=0)

    frames = []
    print("Recording gameplay (1000 steps)...")

    for i in range(1000):
        # AI가 행동 결정
        actions, values, states, neglogpacs = model.step(obs)
        
        # [핵심 수정] AssertionError 해결: actions가 [7] 형태이므로 actions[0]인 7(스칼라)을 전달
        # env_raw.step은 단일 환경이므로 배치가 아닌 단일 액션 값을 원합니다.
        obs, rewards, dones, infos = env_raw.step(actions[0])
        obs = np.expand_dims(obs, axis=0)

        # 화면 캡처
        frame = env_raw.render(mode="rgb_array")
        frames.append(frame)

        if dones:
            print(f"Step {i}: Episode Finished. Resetting...")
            obs = env_raw.reset()
            obs = np.expand_dims(obs, axis=0)

    # 8. 영상 저장
    video_path = 'final_god_ai_result.mp4'
    imageio.mimsave(video_path, frames, fps=20)
    print(f"\nVideo saved to: {os.path.abspath(video_path)}")

finally:
    sess.close()
    display.stop()