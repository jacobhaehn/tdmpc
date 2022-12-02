import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import torch
import numpy as np
import gym
gym.logger.set_level(40)
import time
import random
from pathlib import Path
from cfg import parse_cfg, parse_cfg_atari
from env import make_env, make_env_atari
from algorithm.tdmpc import TDMPC
from algorithm.helper import Episode, ReplayBuffer
import logger
import gym
torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = 'cfgs', 'logs'


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def evaluate(env, agent, num_episodes, step, env_step, video):
	"""Evaluate a trained agent and optionally save a video."""
	episode_rewards = []
	for i in range(num_episodes):
		obs, done, ep_reward, t = env.reset().reshape(3,84,84), False, 0, 0
		if video: video.init(env, enabled=(i==0))
		while not done:
			
			action = agent.plan(obs, eval_mode=True, step=step, t0=t==0)
			action_max = np.argmax(action.cpu().numpy())
			obs, reward, done, _ = env.step(action_max)
			print("I'm Working!, Reward = ", reward)
			obs=obs.reshape(3,84,84)
			ep_reward += reward
			if video: video.record(env)
	
			t += 1
		episode_rewards.append(ep_reward)
		if video: video.save(env_step)
	return np.nanmean(episode_rewards)
	

def train(cfg):
	"""Training script for TD-MPC. Requires a CUDA-enabled device."""
	assert torch.cuda.is_available()
	set_seed(cfg.seed)
	work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
	env, agent, buffer = make_env_atari(cfg), TDMPC(cfg), ReplayBuffer(cfg)
	
	# Run training
	L = logger.Logger(work_dir, cfg)
	episode_idx, start_time = 0, time.time()
	for step in range(0, cfg.train_steps+cfg.episode_length, cfg.episode_length):
		print("Step:", step)
		# Collect trajectory
		obs = env.reset().reshape(3,84,84)
		print(obs.shape)
		episode = Episode(cfg, obs)
		while not episode.done:
			print(len(episode))
			action = agent.plan(obs, step=step, t0=episode.first)
			#print(action)
			# TODO: MAKE SURE THIS MAKES SENSE LATER, Try magnitude vs absolute
			action_max = np.argmax(action.cpu().numpy())
			print("Action_Max = ", action_max)
			#obs, reward, done, _ = env.step(action.cpu().numpy())
			obs, reward, done, _ = env.step(action_max)
			obs=obs.reshape(3,84,84)
			#episode += (obs, action, reward, done)
			episode += (obs, action_max, reward, done)
		#assert len(episode) == cfg.episode_length
		print("episode obs_shape", episode.obs.shape)
		print("episode buffer", buffer._obs.shape)
		buffer += episode

		# Update model
		train_metrics = {}
		if step >= cfg.seed_steps:
			num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
			for i in range(num_updates):
				train_metrics.update(agent.update(buffer, step+i))

		# Log training episode
		episode_idx += 1
		env_step = int(step*cfg.action_repeat)
		common_metrics = {
			'episode': episode_idx,
			'step': step,
			'env_step': env_step,
			'total_time': time.time() - start_time,
			'episode_reward': episode.cumulative_reward}
		train_metrics.update(common_metrics)
		L.log(train_metrics, category='train')

		# Evaluate agent periodically
		if env_step % cfg.eval_freq == 0:
			common_metrics['episode_reward'] = evaluate(env, agent, cfg.eval_episodes, step, env_step, L.video)
			L.log(common_metrics, category='eval')

	L.finish(agent)
	print('Training completed successfully')


if __name__ == '__main__':
	train(parse_cfg_atari(Path().cwd() / __CONFIG__))
