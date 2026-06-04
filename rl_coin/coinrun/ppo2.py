"""
This is a copy of PPO from openai/baselines (https://github.com/openai/baselines/blob/52255beda5f5c8760b0ae1f676aa656bb1a61f80/baselines/ppo2/ppo2.py) with some minor changes.
"""

import time
import joblib
import numpy as np
import tensorflow as tf
from collections import deque
import csv
import os

from mpi4py import MPI

from coinrun.tb_utils import TB_Writer
import coinrun.main_utils as utils

from coinrun.config import Config

mpi_print = utils.mpi_print

from baselines.common.runners import AbstractEnvRunner
from baselines.common.tf_util import initialize
from baselines.common.mpi_util import sync_from_root

class MpiAdamOptimizer(tf.train.AdamOptimizer):
    """Adam optimizer that averages gradients across mpi processes."""
    def __init__(self, comm, **kwargs):
        self.comm = comm
        self.train_frac = 1.0 - Config.get_test_frac()
        tf.train.AdamOptimizer.__init__(self, **kwargs)
    def compute_gradients(self, loss, var_list, **kwargs):
        grads_and_vars = tf.train.AdamOptimizer.compute_gradients(self, loss, var_list, **kwargs)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]

        flat_grad = tf.concat([tf.reshape(g, (-1,)) for g, v in grads_and_vars], axis=0)

        if Config.is_test_rank():
            flat_grad = tf.zeros_like(flat_grad)

        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        num_tasks = self.comm.Get_size()
        buf = np.zeros(sum(sizes), np.float32)

        def _collect_grads(flat_grad):
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks) * self.train_frac, out=buf)
            return buf

        avg_flat_grad = tf.py_func(_collect_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                    for g, (_, v) in zip(avg_grads, grads_and_vars)]

        return avg_grads_and_vars

class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm):
        sess = tf.get_default_session()

        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps)
        norm_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1)

        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        params = tf.trainable_variables()
        weight_params = [v for v in params if '/b' not in v.name]

        total_num_params = 0

        for p in params:
            shape = p.get_shape().as_list()
            num_params = np.prod(shape)
            mpi_print('param', p, num_params)
            total_num_params += num_params

        mpi_print('total num params:', total_num_params)

        l2_loss = tf.reduce_sum([tf.nn.l2_loss(v) for v in weight_params])

        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef + l2_loss * Config.L2_WEIGHT

        if Config.SYNC_FROM_ROOT:
            trainer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=LR, epsilon=1e-5)
        else:
            trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)

        grads_and_var = trainer.compute_gradients(loss, params)

        grads, var = zip(*grads_and_var)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))

        _train = trainer.apply_gradients(grads_and_var)

        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
            advs = returns - values

            adv_mean = np.mean(advs, axis=0, keepdims=True)
            adv_std = np.std(advs, axis=0, keepdims=True)
            advs = (advs - adv_mean) / (adv_std + 1e-8)

            td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr,
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, l2_loss, _train],
                td_map
            )[:-1]
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac', 'l2_loss']

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load

        if Config.SYNC_FROM_ROOT:
            if MPI.COMM_WORLD.Get_rank() == 0:
                initialize()
            
            global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
            sync_from_root(sess, global_variables) #pylint: disable=E1101
        else:
            initialize()

class Runner(AbstractEnvRunner):
    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.lam = lam
        self.gamma = gamma

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for i, info in enumerate(infos):
                maybeepinfo = info.get('episode')
                if maybeepinfo and self.dones[i]: 
                    epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def constfn(val):
    def f(_):
        return val
    return f


def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, load_path=None,
            eval_env=None, eval_interval=1000000):

    #///////////////
    # --- [연구원님 맞춤형 데이터 추출 세팅] ---
    log_dir = "tracked_metrics"
    os.makedirs(log_dir, exist_ok=True)

    # 1. Loss 저장용 CSV
    loss_file = open(os.path.join(log_dir, "losses.csv"), "w", newline="")
    loss_writer = csv.writer(loss_file)
    loss_writer.writerow(["timestep", "policy_loss", "value_loss", "policy_entropy", "approxkl", "clipfrac", "l2_loss"])

    # 2. 리워드 빈도 저장용 CSV (누적 및 100만 구간)
    reward_file = open(os.path.join(log_dir, "rewards_and_reasons.csv"), "w", newline="")
    reward_writer = csv.writer(reward_file)
    reward_writer.writerow(["timestep", "type", "reason", "count"]) # type: cumulative / interval_1M

    # 3. 에피소드 끝나는 이유별 평균 길이 및 보상 저장용 CSV
    episode_file = open(os.path.join(log_dir, "episode_stats_by_reason.csv"), "w", newline="")
    ep_writer = csv.writer(episode_file)
    ep_writer.writerow(["timestep", "reason", "avg_length", "avg_reward"])

    # 4. 10번 에피소드당 평균 reward 저장용 CSV
    recent_10_file = open(os.path.join(log_dir, "recent_update_rewards.csv"), "w", newline="")
    recent_10_writer = csv.writer(recent_10_file)
    recent_10_writer.writerow(["timestep", "avg_reward_1update"])
    
    # --- 💡 [추가] 5. Validation 결과 저장용 CSV ---
    if eval_env is not None:
        val_file = open(os.path.join(log_dir, "validation_stats.csv"), "w", newline="")
        val_writer = csv.writer(val_file)
        val_writer.writerow(["timestep", "val_avg_reward", "val_success_rate"])
        next_val_checkpoint = eval_interval
    # -----------------------------------------------

    # 내부 카운터 및 버퍼 초기화
    recent_10_rewards = []
    
    # dummy_info 예시 후보들 ('coin', 'death', 'fall', 'timeout' 등)
    cumulative_counts = {}
    interval_counts = {}
    episode_lengths_by_reason = {}
    episode_rewards_by_reason = {}

    next_1m_checkpoint = 1000000
    # ----------------------------------------
    #/////////////////


    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    sess = tf.get_default_session()
    tb_writer = TB_Writer(sess)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    
    nbatch_train = nbatch // nminibatches

    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm)

    utils.load_all_params(sess)

    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    epinfobuf10 = deque(maxlen=10)
    epinfobuf100 = deque(maxlen=100)
    tfirststart = time.time()
    active_ep_buf = epinfobuf100

    nupdates = total_timesteps//nbatch
    mean_rewards = []
    datapoints = []

    run_t_total = 0
    train_t_total = 0

    can_save = True
    checkpoints = [32, 64]
    saved_key_checkpoints = [False] * len(checkpoints)

    if Config.SYNC_FROM_ROOT and rank != 0:
        can_save = False

    def save_model(base_name=None):
        base_dict = {'datapoints': datapoints}
        utils.save_params_in_scopes(sess, ['model'], Config.get_save_file(base_name=base_name), base_dict)

    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)

        mpi_print('collecting rollouts...')
        run_tstart = time.time()

        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()
        # ////////////////////////
        # --- [에피소드 종료 원인 및 리워드 분석 분석 로직] ---
        current_step = update * nbatch  # 💡 추가: 현재 몇 번째 스텝인지 계산f
        
        for epinfo in epinfos:
            r = epinfo.get('r', 0) # 에피소드 총 보상
            l = epinfo.get('l', 0) # 에피소드 길이
            
            # 💡 [수정됨] C++ dummy_info 없이 점수와 길이만으로 원인 추론하기
            if r > 0:
                reason = 'success_coin'   # 코인을 먹고 깸
            elif l >= 1000:
                reason = 'timeout'        # 1000스텝 동안 코인 못 먹고 시간 초과
            else:
                reason = 'death'          # 1000스텝 전에 점수 없이 끝남 (톱니바퀴, 괴물, 낙사)

            if reason not in cumulative_counts:
                cumulative_counts[reason] = 0
                interval_counts[reason] = 0
                episode_lengths_by_reason[reason] = []
                episode_rewards_by_reason[reason] = []

            cumulative_counts[reason] += 1
            interval_counts[reason] += 1
            episode_lengths_by_reason[reason].append(l)
            episode_rewards_by_reason[reason].append(r)

            recent_10_rewards.append(r)

        if len(recent_10_rewards) > 0:
            batch_avg = sum(recent_10_rewards) / len(recent_10_rewards)
            recent_10_writer.writerow([current_step, batch_avg])
            recent_10_file.flush()
            recent_10_rewards.clear()
        # ///////////////////////
        # --------------------------------------------------
        epinfobuf10.extend(epinfos)
        epinfobuf100.extend(epinfos)

        run_elapsed = time.time() - run_tstart
        run_t_total += run_elapsed
        mpi_print('rollouts complete')

        mblossvals = []

        mpi_print('updating parameters...')
        train_tstart = time.time()

        if states is None: # nonrecurrent version
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        else: # recurrent version
            assert nenvs % nminibatches == 0
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        # update the dropout mask
        sess.run([model.train_model.dropout_assign_ops])

        train_elapsed = time.time() - train_tstart
        train_t_total += train_elapsed
        mpi_print('update complete')

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))

        if update % log_interval == 0 or update == 1:
            step = update*nbatch
            rew_mean_10 = utils.process_ep_buf(active_ep_buf, tb_writer=tb_writer, suffix='', step=step)
            ep_len_mean = np.nanmean([epinfo['l'] for epinfo in active_ep_buf])
            
            mpi_print('\n----', update)

            mean_rewards.append(rew_mean_10)
            datapoints.append([step, rew_mean_10])

            tb_writer.log_scalar(ep_len_mean, 'ep_len_mean')
            tb_writer.log_scalar(fps, 'fps')

            mpi_print('time_elapsed', tnow - tfirststart, run_t_total, train_t_total)
            mpi_print('timesteps', update*nsteps, total_timesteps)

            mpi_print('eplenmean', ep_len_mean)
            mpi_print('eprew', rew_mean_10)
            mpi_print('fps', fps)
            mpi_print('total_timesteps', update*nbatch)
            mpi_print([epinfo['r'] for epinfo in epinfobuf10])

            if len(mblossvals):
                for (lossval, lossname) in zip(lossvals, model.loss_names):
                    mpi_print(lossname, lossval)
                    tb_writer.log_scalar(lossval, lossname)
            mpi_print('----\n')
            # ///////////
            if len(lossvals) >= 6: 
                # [step] 뒤에 lossvals 리스트의 6개 항목을 쫙 풀어서 연결해줍니다.
                loss_writer.writerow([step] + list(lossvals))
                loss_file.flush()
            # ///////////

        if can_save:
            if save_interval and (update % save_interval == 0):
                save_model()

            for j, checkpoint in enumerate(checkpoints):
                if (not saved_key_checkpoints[j]) and (step >= (checkpoint * 1e6)):
                    saved_key_checkpoints[j] = True
                    save_model(str(checkpoint) + 'M')

        # --- [100만 타임스텝 체크포인트 정산 (누락됐던 부분!)] ---
        current_step = update * nbatch
        if current_step >= next_1m_checkpoint:
            for reason_key in cumulative_counts.keys():
                # 누적 빈도 및 100만번 마다의 개별 빈도 기록
                reward_writer.writerow([next_1m_checkpoint, "cumulative", reason_key, cumulative_counts[reason_key]])
                reward_writer.writerow([next_1m_checkpoint, "interval_1M", reason_key, interval_counts[reason_key]])
                
                # 끝나는 이유별 에피소드 평균 길이 및 보상 정산
                lengths = episode_lengths_by_reason[reason_key]
                rewards = episode_rewards_by_reason[reason_key]
                avg_l = sum(lengths) / len(lengths) if lengths else 0
                avg_r = sum(rewards) / len(rewards) if rewards else 0
                ep_writer.writerow([next_1m_checkpoint, reason_key, avg_l, avg_r])
            
            reward_file.flush()
            episode_file.flush()

            # 100만번 개별 구간 카운터 및 버퍼 리스트 비우기 (누적은 유지)
            for reason_key in interval_counts.keys():
                interval_counts[reason_key] = 0
                episode_lengths_by_reason[reason_key] = []
                episode_rewards_by_reason[reason_key] = []
                
            next_1m_checkpoint += 1000000

        if eval_env is not None and current_step >= next_val_checkpoint:
            mpi_print(f"\n>>> Running Validation at {current_step} steps... <<<")
            
            val_obs = eval_env.reset()
            val_states = model.initial_state
            val_dones = [False for _ in range(eval_env.num_envs)]
            
            val_ep_rewards = []
            val_ep_successes = []
            
            # 평가용으로 총 20개의 에피소드를 플레이합니다.
            while len(val_ep_rewards) < 20:
                
                # --- 💡 [Shape 에러 해결] 1개짜리 데이터를 128개(nenvs) 크기로 뻥튀기(Padding) ---
                # 1. 128개짜리 빈 껍데기(0)를 만들고, 첫 번째 칸에만 진짜 화면을 넣습니다.
                padded_obs = np.zeros((nenvs,) + val_obs.shape[1:], dtype=val_obs.dtype)
                padded_obs[0] = val_obs[0]
                
                # 2. 게임 종료(Done) 상태도 128개짜리로 맞춥니다.
                padded_dones = np.zeros(nenvs, dtype=np.bool)
                padded_dones[0] = val_dones[0]
                
                # 3. 훈련된 모델에 128개 묶음을 통째로 넣고 행동을 예측합니다.
                padded_actions, _, val_states, _ = model.step(padded_obs, val_states, padded_dones)
                
                # 4. 128개의 행동 결과 중, 진짜 화면에 대한 '첫 번째 행동'만 쏙 잘라냅니다.
                val_actions = padded_actions[0:1]
                # -------------------------------------------------------------------------
                
                val_obs, val_rewards, val_dones, val_infos = eval_env.step(val_actions)
                
                for i, info in enumerate(val_infos):
                    maybeepinfo = info.get('episode')
                    if maybeepinfo and val_dones[i]:
                        r = maybeepinfo['r']
                        val_ep_rewards.append(r)
                        val_ep_successes.append(1 if r > 0 else 0) # 리워드가 0보다 크면 클리어(성공)로 간주
                        
            val_avg_reward = np.mean(val_ep_rewards)
            val_success_rate = np.mean(val_ep_successes)
            
            mpi_print(f">>> Validation Result - Avg Reward: {val_avg_reward:.3f} | Success Rate: {val_success_rate*100:.1f}% <<<")
            
            # CSV에 기록
            val_writer.writerow([current_step, val_avg_reward, val_success_rate])
            val_file.flush()
            
            # 다음 평가 시점 업데이트
            next_val_checkpoint += eval_interval
        # ----------------------------------------

    save_model()

    env.close()
    return mean_rewards
