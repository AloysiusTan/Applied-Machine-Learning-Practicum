#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from PIL import Image
from tf_agents.environments import suite_gym, tf_py_environment, py_environment, TFPyEnvironment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn import dqn_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.metrics import tf_metrics
from tf_agents.utils.common import function
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.eval.metric_utils import log_metrics
from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.trajectories.trajectory import to_transition
import cv2
import gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tf_agents.trajectories.trajectory import to_transition

import logging
logging.basicConfig(level=logging.INFO)

# Increase the embed limit for animations
plt.rcParams['animation.embed_limit'] = 50 * 1024 * 1024  # 50 MB


# In[2]:


# Set up environment
env_name = "QbertNoFrameskip-v4"
max_episode_steps = 10000  # <=> 108k ALE frames since 1 step = 4 frames

env = suite_atari.load(
    env_name,
    max_episode_steps=max_episode_steps,
    gym_env_wrappers=[AtariPreprocessing, FrameStack4]
)
tf_env = tf_py_environment.TFPyEnvironment(env)


# In[3]:


max_episode_steps = 10000  # <=> 108k ALE frames since 1 step = 4 frames
environment_name = "QbertNoFrameskip-v4"

class AtariPreprocessingForQbert(AtariPreprocessing):
    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        return obs

    def step(self, action):
        lives_before_action = self.ale.lives()
        obs, rewards, done, info = super().step(action)
        return obs, rewards, done, info

# Show progress class
class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total

    def __call__(self, trajectories):
        steps = trajectories.step_type.shape[0]
        self.counter += steps
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")


# In[4]:


def render_policy_net(model, n_max_steps=200, seed=42):
    frames = []
    env = gym.make(env_name)
    env.seed(seed)
    np.random.seed(seed)
    obs = env.reset()
    for step in range(n_max_steps):
        frames.append(env.render(mode="rgb_array"))
        q_values = model.predict(obs[np.newaxis])
        action = np.argmax(q_values[0])
        obs, reward, done, info = env.step(action)
        if done:
            break
    env.close()
    return frames

def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.close()
    return anim

def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    
def plot_observation(obs):
    # Since there are only 3 color channels, you cannot display 4 frames
    # with one primary color per frame. So this code computes the delta between
    # the current frame and the mean of the other frames, and it adds this delta
    # to the red and blue channels to get a pink color for the current frame.
    obs = obs.astype(np.float32)
    img = obs[..., :3]
    current_frame_delta = np.maximum(obs[..., 3] - obs[..., :3].mean(axis=-1), 0.)
    img[..., 0] += current_frame_delta
    img[..., 2] += current_frame_delta
    img = np.clip(img / 150, 0, 1)
    plt.imshow(img)
    plt.axis("off")


# In[5]:


# Learning rate schedule
learning_rate_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=2.5e-4,
    decay_steps=100000,
    end_learning_rate=1e-5
)

# Optimizer
optimizer = tf.keras.optimizers.RMSprop(
    learning_rate=learning_rate_schedule,
    rho=0.95, momentum=0.0, epsilon=0.00001, centered=True
)

# DQN Model Architecture for Qbert
preprocessing_layer = tf.keras.layers.Lambda(lambda obs: tf.cast(obs, np.float32) / 255.)
conv_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
fc_layer_params = [512]

q_net = QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    preprocessing_layers=preprocessing_layer,
    conv_layer_params=conv_layer_params,
    fc_layer_params=fc_layer_params
)

# DQN Agent
train_step = tf.Variable(0)
update_period = 4
agent = dqn_agent.DqnAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    target_update_period=2000,
    td_errors_loss_fn=tf.keras.losses.Huber(reduction="none"),
    gamma=0.99,
    train_step_counter=train_step,
    epsilon_greedy=lambda: tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=1.0,
        decay_steps=250000 // update_period,
        end_learning_rate=0.01)(train_step)
)
agent.initialize()

# Replay buffer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=10000
)

replay_buffer_observer = replay_buffer.add_batch


# In[6]:


# Checkpoint directory
checkpoint_dir = "./checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "checkpoint")

# Create checkpoint manager
checkpoint = tf.train.Checkpoint(agent=agent, optimizer=optimizer, train_step=train_step)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=10)

# Restore from the latest checkpoint if available
if checkpoint_manager.latest_checkpoint:
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    print(f"Restored from {checkpoint_manager.latest_checkpoint}")
    print(f"Start from step {int(train_step)}")
else:
    print("Starting from scratch.")


# In[7]:


train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric(),
]

collect_driver = DynamicStepDriver(
    tf_env,
    agent.collect_policy,
    observers=[replay_buffer_observer] + train_metrics,
    num_steps=update_period) # collect 4 steps for each training iteration

tf.random.set_seed(9) # chosen to show an example of trajectory at the end of an episode
dataset = replay_buffer.as_dataset(
    sample_batch_size=64,
    num_steps=2,
    num_parallel_calls=3).prefetch(3)

collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)


# In[10]:


# Train agent function
def resume_training(n_iterations):
    print(f"Resuming training from step: {int(train_step)}")
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    iterator = iter(dataset)
    for iteration in range(n_iterations):
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)
        print("\r{} loss:{:.5f}".format(iteration, train_loss.loss.numpy()), end="")
        if iteration % 1000 == 0:
            log_metrics(train_metrics)
        if iteration % (n_iterations // 10) == 0:
            checkpoint_manager.save()
            print(f"\nCheckpoint saved at iteration {iteration}")
            log_metrics(train_metrics)


# In[11]:


resume_training(50_000)

