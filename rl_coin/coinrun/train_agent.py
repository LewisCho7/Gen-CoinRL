"""
Train an agent using a PPO2 based on OpenAI Baselines.
"""

import time
from mpi4py import MPI
import tensorflow as tf
from baselines.common import set_global_seeds
import coinrun.main_utils as utils
from coinrun import setup_utils, policies, wrappers, ppo2
from coinrun.config import Config

def main():
    args = setup_utils.setup_and_load()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    seed = int(time.time()) % 10000
    set_global_seeds(seed * 100 + rank)

    utils.setup_mpi_gpus()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True 
    #config.gpu_options.per_process_gpu_memory_fraction = 0.3  # GPU 메모리의 40%만 사용하도록 엄격히 제한
    #config.allow_soft_placement = True                        # 특정 연산이 GPU에서 안 될 경우 CPU로 자동 전환
    #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.OFF

    nenvs = Config.NUM_ENVS
    total_timesteps = int(256000000)
    save_interval = args.save_interval

    env = utils.make_general_env(nenvs, seed=rank)

    with tf.Session(config=config):
        env = wrappers.add_final_wrappers(env)
        
        policy = policies.get_policy()

        ppo2.learn(policy=policy,
                    env=env,
                    save_interval=save_interval,
                    nsteps=Config.NUM_STEPS,
                    nminibatches=Config.NUM_MINIBATCHES,
                    lam=0.95,
                    gamma=Config.GAMMA,
                    noptepochs=Config.PPO_EPOCHS,
                    log_interval=1,
                    ent_coef=Config.ENTROPY_COEFF,
                    lr=lambda f : f * Config.LEARNING_RATE,
                    cliprange=lambda f : f * 0.2,
                    total_timesteps=total_timesteps)

if __name__ == '__main__':
    main()

