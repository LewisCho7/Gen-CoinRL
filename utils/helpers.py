# utils/helpers.py
import random
import os
import numpy as np
import torch

def set_seed(seed: int):
    """
    실험 재현성을 위해 프로젝트 전체의 시드(Seed)를 고정하는 함수
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 멀티 GPU 사용할 경우
    
    # 파이토치 내부 연산 알고리즘 고정 (학습 속도가 약간 느려질 수 있음)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"🔒 모든 난수 시드가 [{seed}]로 고정되었습니다.")

def print_args(args):
    """
    입력받은 명령어 인자(Arguments)를 터미널에 보기 좋게 출력하는 함수
    """
    print("\n" + "="*40)
    print(f"{'[ Experiment Configuration ]':^40}")
    print("="*40)
    for arg, value in vars(args).items():
        print(f" • {arg:<15} : {value}")
    print("="*40 + "\n")
