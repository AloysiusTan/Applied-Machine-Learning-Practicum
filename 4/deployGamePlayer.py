#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tf_agents.environments import suite_gym, suite_atari, tf_py_environment
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.replay_buffers import tf_uniform_replay_buffer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import Image as IPImage

# Increase the embed limit for animations
plt.rcParams['animation.embed_limit'] = 50 * 1024 * 1024  # 50 MB


# In[2]:


# Set up the environment
env_name = "QbertNoFrameskip-v4"
max_episode_steps = 10000  # <=> 108k ALE frames since 1 step = 4 frames

env = suite_atari.load(
    env_name,
    max_episode_steps=max_episode_steps,
    gym_env_wrappers=[AtariPreprocessing, FrameStack4]
)
tf_env = tf_py_environment.TFPyEnvironment(env)

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


# In[3]:


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
else:
    print("Starting from scratch.")


# In[4]:


# Function to create and save gameplay video
def create_policy_eval_video(policy, filename, num_episodes=5, fps=30):
    frames = []
    for _ in range(num_episodes):
        time_step = tf_env.reset()
        frames.append(tf_env.pyenv.envs[0].render(mode='rgb_array'))
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = tf_env.step(action_step.action)
            frames.append(tf_env.pyenv.envs[0].render(mode='rgb_array'))
    frame_images = [Image.fromarray(frame) for frame in frames]
    frame_images[0].save(filename, format='GIF',
                         append_images=frame_images[1:],
                         save_all=True,
                         duration=1000//fps, loop=0)
    return frames


# In[5]:


def update_scene(num, frames, patch):
    patch.set_data(frames[num])

def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.show()


# In[6]:


# Create and save the evaluation video
video_filename = "videos/qbert_gameplay.gif"
frames = create_policy_eval_video(agent.policy, video_filename)
print(f"Evaluation video saved as {video_filename}")


# In[7]:


# Display the video
IPImage(filename=video_filename)

