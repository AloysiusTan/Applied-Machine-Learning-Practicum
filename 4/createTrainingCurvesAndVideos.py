#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import os
from PIL import Image as PILImage
from tf_agents.environments import suite_gym, suite_atari, tf_py_environment
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
import matplotlib.pyplot as plt
import re
from IPython.display import Image, display


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

# Define the Q-network
input_shape = (84, 84, 4) 
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

# Define the DQN agent
train_step = tf.Variable(0)
agent = dqn_agent.DqnAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    q_network=q_net,
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=2.5e-4, rho=0.95, momentum=0.0, epsilon=0.00001, centered=True),
    target_update_period=2000,
    td_errors_loss_fn=tf.keras.losses.Huber(reduction="none"),
    gamma=0.99,
    train_step_counter=train_step
)
agent.initialize()


# In[3]:


# Restore from the latest checkpoint if available
checkpoint_dir = "./checkpoints"
checkpoint = tf.train.Checkpoint(agent=agent, train_step=train_step)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=10)

# Function to create and save evaluation video
def create_policy_eval_video(policy, filename, num_episodes=5, fps=30):
    frames = []
    for _ in range(num_episodes):
        time_step = tf_env.reset()
        frames.append(tf_env.pyenv.envs[0].render(mode='rgb_array'))
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = tf_env.step(action_step.action)
            frames.append(tf_env.pyenv.envs[0].render(mode='rgb_array'))
    frame_images = [PILImage.fromarray(frame) for frame in frames]
    frame_images[0].save(filename, format='GIF',
                         append_images=frame_images[1:],
                         save_all=True,
                         duration=1000//fps, loop=0)

# Ensure the videos directory exists
videos_dir = "./videos"
os.makedirs(videos_dir, exist_ok=True)


# In[4]:


# Helper function to extract the numeric part from the checkpoint filename
def extract_checkpoint_number(checkpoint_filename):
    match = re.search(r'\d+', checkpoint_filename)
    if match:
        return int(match.group())
    return -1

# Load checkpoints and create evaluation videos
checkpoint_files = sorted(os.listdir(checkpoint_dir), key=lambda x: extract_checkpoint_number(x))
for checkpoint_file in checkpoint_files:
    if "index" in checkpoint_file:
        checkpoint_prefix = checkpoint_file.split(".")[0]
        status = checkpoint.restore(os.path.join(checkpoint_dir, checkpoint_prefix))
        print(f"Restored from {checkpoint_prefix}")
        
        # Create and save the evaluation video
        video_path = os.path.join(videos_dir, f"{checkpoint_prefix}_eval.gif")
        create_policy_eval_video(agent.policy, video_path)
        print(f"Evaluation video saved for {checkpoint_prefix}")

print("All evaluation videos saved.")


# In[5]:


# Load checkpoints and create evaluation videos
video_filenames = sorted(
    [f for f in os.listdir(videos_dir)  if f.startswith('ckpt') and f.endswith('.gif')],
    key=lambda x: extract_checkpoint_number(x)
)

# Display the videos
for video in video_filenames:
    print(video)
    display(Image(filename=os.path.join(videos_dir, video)))

