# Copyright 2020 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from typing import Optional, Callable, Union, Tuple, List

import keras
import numpy as np
import random
import os
import json
import tensorflow.compat.v2 as tf
tf.compat.v1.enable_v2_behavior()
import pickle

from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment

from dice_rl.environments.env_policies import get_target_policy
import dice_rl.environments.gridworld.navigation as navigation
import dice_rl.environments.gridworld.tree as tree
import dice_rl.environments.gridworld.taxi as taxi
from dice_rl.estimators.neural_dice import NeuralDice
from dice_rl.estimators import estimator as estimator_lib
from dice_rl.networks.value_network import ValueNetwork
import dice_rl.utils.common as common_utils
from dice_rl.data.dataset import Dataset, EnvStep, StepType
from dice_rl.data.tf_offpolicy_dataset import TFOffpolicyDataset


FLAGS = flags.FLAGS

flags.DEFINE_string('load_dir', None, 'Directory to load dataset from.')
flags.DEFINE_string('save_dir', None,
                    'Directory to save the model and estimation results.')
flags.DEFINE_string('env_name', 'grid', 'Environment name.')
flags.DEFINE_integer('seed', 0, 'Initial random seed.')
flags.DEFINE_bool('tabular_obs', False, 'Whether to use tabular observations.')
flags.DEFINE_integer('num_trajectory', 1000,
                     'Number of trajectories to collect.')
flags.DEFINE_integer('max_trajectory_length', 40,
                     'Cutoff trajectory at this step.')
flags.DEFINE_float('alpha', 0.0, 'How close to target policy.')
flags.DEFINE_float('gamma', 0.99, 'Discount factor.')
flags.DEFINE_float('nu_learning_rate', 0.0001, 'Learning rate for nu.')
flags.DEFINE_float('zeta_learning_rate', 0.0001, 'Learning rate for zeta.')
flags.DEFINE_float('nu_regularizer', 0.0, 'Ortho regularization on nu.')
flags.DEFINE_float('zeta_regularizer', 0.0, 'Ortho regularization on zeta.')
flags.DEFINE_integer('num_steps', 100000, 'Number of training steps.')
flags.DEFINE_integer('batch_size', 2048, 'Batch size.')

flags.DEFINE_float('f_exponent', 2, 'Exponent for f function.')
flags.DEFINE_bool('primal_form', False,
                  'Whether to use primal form of loss for nu.')

flags.DEFINE_float('primal_regularizer', 0.,
                   'LP regularizer of primal variables.')
flags.DEFINE_float('dual_regularizer', 1., 'LP regularizer of dual variables.')
flags.DEFINE_bool('zero_reward', False,
                  'Whether to ignore reward in optimization.')
flags.DEFINE_float('norm_regularizer', 1.,
                   'Weight of normalization constraint.')
flags.DEFINE_bool('zeta_pos', True, 'Whether to enforce positivity constraint.')

flags.DEFINE_float('scale_reward', 1., 'Reward scaling factor.')
flags.DEFINE_float('shift_reward', 0., 'Reward shift factor.')
flags.DEFINE_string(
    'transform_reward', None, 'Non-linear reward transformation'
    'One of [exp, cuberoot, None]')

flags.DEFINE_integer('fold_number', 2, 'Number of folds for cross-fitting.')


def main(argv):
  load_dir = FLAGS.load_dir
  save_dir = FLAGS.save_dir
  env_name = FLAGS.env_name
  seed = FLAGS.seed
  tabular_obs = FLAGS.tabular_obs
  num_trajectory = FLAGS.num_trajectory
  max_trajectory_length = FLAGS.max_trajectory_length
  alpha = FLAGS.alpha
  gamma = FLAGS.gamma
  nu_learning_rate = FLAGS.nu_learning_rate
  zeta_learning_rate = FLAGS.zeta_learning_rate
  nu_regularizer = FLAGS.nu_regularizer
  zeta_regularizer = FLAGS.zeta_regularizer
  num_steps = FLAGS.num_steps
  batch_size = FLAGS.batch_size

  f_exponent = FLAGS.f_exponent
  primal_form = FLAGS.primal_form

  primal_regularizer = FLAGS.primal_regularizer
  dual_regularizer = FLAGS.dual_regularizer
  zero_reward = FLAGS.zero_reward
  norm_regularizer = FLAGS.norm_regularizer
  zeta_pos = FLAGS.zeta_pos

  scale_reward = FLAGS.scale_reward
  shift_reward = FLAGS.shift_reward
  transform_reward = FLAGS.transform_reward

  fold_number = FLAGS.fold_number

  def reward_fn(env_step):
    reward = env_step.reward * scale_reward + shift_reward
    if transform_reward is None:
      return reward
    if transform_reward == 'exp':
      reward = tf.math.exp(reward)
    elif transform_reward == 'cuberoot':
      reward = tf.sign(reward) * tf.math.pow(tf.abs(reward), 1.0 / 3.0)
    else:
      raise ValueError('Reward {} not implemented.'.format(transform_reward))
    return reward

  hparam_str = ('{ENV_NAME}_tabular{TAB}_alpha{ALPHA}_seed{SEED}_'
                'numtraj{NUM_TRAJ}_maxtraj{MAX_TRAJ}').format(
                    ENV_NAME=env_name,
                    TAB=tabular_obs,
                    ALPHA=alpha,
                    SEED=seed,
                    NUM_TRAJ=num_trajectory,
                    MAX_TRAJ=max_trajectory_length)
  hparam_str_target = ('{ENV_NAME}_tabular{TAB}_alpha1.0_seed{SEED}_'
                  'numtraj{NUM_TRAJ}_maxtraj{MAX_TRAJ}').format(
                    ENV_NAME=env_name,
                    TAB=tabular_obs,
                    SEED=seed,
                    NUM_TRAJ=num_trajectory,
                    MAX_TRAJ=max_trajectory_length)
  train_hparam_str = (
      'nlr{NLR}_zlr{ZLR}_zeror{ZEROR}_preg{PREG}_dreg{DREG}_nreg{NREG}_'
      'pform{PFORM}_fexp{FEXP}_zpos{ZPOS}_'
      'scaler{SCALER}_shiftr{SHIFTR}_transr{TRANSR}').format(
          NLR=nu_learning_rate,
          ZLR=zeta_learning_rate,
          ZEROR=zero_reward,
          PREG=primal_regularizer,
          DREG=dual_regularizer,
          NREG=norm_regularizer,
          PFORM=primal_form,
          FEXP=f_exponent,
          ZPOS=zeta_pos,
          SCALER=scale_reward,
          SHIFTR=shift_reward,
          TRANSR=transform_reward)
  if save_dir is not None:
    save_dir = os.path.join(save_dir, hparam_str, train_hparam_str)
    summary_writer = tf.summary.create_file_writer(logdir=save_dir)
    summary_writer.set_as_default()
  else:
    tf.summary.create_noop_writer()

  directory = os.path.join(load_dir, hparam_str)
  print('Loading dataset from', directory)
  dataset: TFOffpolicyDataset = Dataset.load(directory)
  # dataset.get_data_table_contents("./table_contents_original.json")



  def random_split(train_ratio, k_fold):

    num_samples = (dataset.num_total_steps).numpy()
    num_eval_samples = int(train_ratio * num_samples)

    episodes, valid_steps = dataset.get_all_episodes_1_by_1()
    combined = list(zip(episodes, valid_steps))
    random.shuffle(combined)
    all_episodes, all_valid_steps = zip(*combined)

    folds = []

    if k_fold == 2:

      # Split the dataset into training and evaluation subsets
      train_dataset = TFOffpolicyDataset(
          spec=dataset.spec,
          capacity=num_eval_samples,
          name='TrainDataset',
      )
      eval_dataset = TFOffpolicyDataset(
          spec=dataset.spec,
          capacity=num_samples - num_eval_samples,
          name='EvalDataset',
      )

      split_index = len(all_episodes) // 2
      train_episodes = all_episodes[:split_index]
      train_valid_steps = all_valid_steps[:split_index]
      eval_episodes = all_episodes[split_index:]
      eval_valid_steps = all_valid_steps[split_index:]

      for i, (episode, valid_steps) in enumerate(zip(train_episodes, train_valid_steps)):
        add_episodes_to_dataset(episode, valid_steps, train_dataset)
      
      for i, (episode, valid_steps) in enumerate(zip(eval_episodes, eval_valid_steps)):
        add_episodes_to_dataset(episode, valid_steps, eval_dataset)

      folds.append((train_dataset, eval_dataset))
      folds.append((eval_dataset, train_dataset))

      # train_dataset.get_data_table_contents("./table_contents_train.json")
      # eval_dataset.get_data_table_contents("./table_contents_eval.json")

    elif k_fold == 5: 

      fold_size = len(all_episodes) // k_fold

      for i in range(k_fold):
          start = i * fold_size
          end = (i + 1) * fold_size if i < k_fold - 1 else len(all_episodes)

          eval_episodes = all_episodes[start:end]
          eval_valid_steps = all_valid_steps[start:end]

          train_episodes = all_episodes[:start] + all_episodes[end:]
          train_valid_steps = all_valid_steps[:start] + all_valid_steps[end:]

          train_dataset = TFOffpolicyDataset(
              spec=dataset.spec,
              capacity=num_samples-num_eval_samples,
              name=f'TrainDataset_Fold{i + 1}',
          )
          eval_dataset = TFOffpolicyDataset(
              spec=dataset.spec,
              capacity=num_eval_samples,
              name=f'EvalDataset_Fold{i + 1}',
          )

          for i, (episode, valid_steps) in enumerate(zip(train_episodes, train_valid_steps)):
              add_episodes_to_dataset(episode, valid_steps, train_dataset)
          
          for i, (episode, valid_steps) in enumerate(zip(eval_episodes, eval_valid_steps)):
              add_episodes_to_dataset(episode, valid_steps, eval_dataset)

          folds.append((train_dataset, eval_dataset))

    return folds


  def add_episodes_to_dataset(episodes, valid_ids, write_dataset: TFOffpolicyDataset):
    num_episodes = 1 if tf.rank(valid_ids) == 1 else tf.shape(valid_ids)[0]
    for ep_id in range(num_episodes):
      if tf.rank(valid_ids) == 1:
        this_valid_ids = valid_ids
        this_episode = episodes
      else:
        this_valid_ids = valid_ids[ep_id, ...]
        this_episode = tf.nest.map_structure(
            lambda t: t[ep_id, ...], episodes)

      episode_length = tf.shape(this_valid_ids)[0]
      for step_id in range(episode_length):
        this_valid_id = this_valid_ids[step_id]
        this_step = tf.nest.map_structure(
            lambda t: t[step_id, ...], this_episode)
        if this_valid_id:
          write_dataset.add_step(this_step)

  

  def run_training_and_estimation(estimator : NeuralDice, 
                                  train_dataset : TFOffpolicyDataset, 
                                  eval_dataset : TFOffpolicyDataset,
                                  ground_truth: float,
                                  joint_estimates: Optional[List[float]] = None, 
                                  k_fold: Optional[int] = None):

    global_step = tf.Variable(0, dtype=tf.int64)
    tf.summary.experimental.set_step(global_step)

    target_policy = get_target_policy(load_dir, env_name, tabular_obs)
    running_losses = []
    running_estimates = []

    for step in range(num_steps): 

      transitions_batch = train_dataset.get_step(batch_size, num_steps=2)
      initial_steps_batch, _ = train_dataset.get_episode(batch_size, truncate_episode_at=1)
      initial_steps_batch = tf.nest.map_structure(lambda t: t[:, 0, ...], initial_steps_batch)

      losses = estimator.train_step(initial_steps_batch, transitions_batch, target_policy)
      running_losses.append(losses)

      if step == num_steps - 1:
        estimate = estimator.estimate_average_reward(eval_dataset, target_policy)
        running_estimates.append(estimate)
        joint_estimates = calculate_joint_estimate(running_estimates, joint_estimates, ground_truth, k_fold)
        running_losses = []
      elif step % 100 == 0:
        estimate = estimator.estimate_average_reward(eval_dataset, target_policy)
        running_estimates.append(estimate)
        running_losses = []

      global_step.assign_add(1)
    
    if k_fold is not None:
      return joint_estimates
  
  
  def run_cross_fitting():
    
    # if fold_number == 2:
    #   train_ratio = 0.5
    # elif fold_number == 5:
    #   train_ratio = 0.2

    train_ratio = float(1/fold_number)

    folds_list = random_split(train_ratio, fold_number)

    joint_estimates = []

    target_directory = os.path.join(load_dir, hparam_str_target)
    target_dataset: TFOffpolicyDataset = Dataset.load(target_directory)
    ground_truth = estimator_lib.get_fullbatch_average(target_dataset)

    for fold in range(fold_number):

      train_dataset, eval_dataset = folds_list[fold]
      estimator = build_estimator(train_dataset)

      current_fold = fold+1
      joint_estimates = run_training_and_estimation(estimator, 
                                                    train_dataset, 
                                                    eval_dataset, 
                                                    ground_truth, 
                                                    joint_estimates,
                                                    current_fold)
      
      print(f'Fold {current_fold}: Estimation DONE!')
  
  
  def calculate_joint_estimate(average_rewards: List[float], 
                               joint_estimates: List[float],
                               ground_truth: float, 
                               fold_count):

    joint_estimates.append(average_rewards)
      
    step_indices = list(range(len(average_rewards)))
    step_indices = [index * 100 for index in step_indices]
    # estimator_lib.plot_per_step_reward(step_indices, ground_truth, average_rewards, fold_count)

    if fold_count == fold_number:

      joint_estimate_array = np.array(joint_estimates)
      final_mean = np.mean(joint_estimate_array, axis=0)
      final_mean_list = final_mean.tolist()

      if fold_number == 2:
        filename = "./results/grid/log_fold2.json"
      elif fold_number == 5:
        filename = "./results/grid/log_fold5.json"
      
      # Convert EagerTensors to lists or other serializable types
      step_indices_list = step_indices.tolist() if hasattr(step_indices, 'tolist') else step_indices
      joint_estimates_list = [je.numpy().tolist() if hasattr(je, 'numpy') else je for je in final_mean_list]
      ground_truth_list = ground_truth.numpy().tolist() if hasattr(ground_truth, 'numpy') else ground_truth

      # Read existing data from the JSON file
      if os.path.exists(filename):
          with open(filename, 'r') as file:
              data = json.load(file)
      else:
          data = {"steps": step_indices_list, "rewards_list": [], "ground_truth": []}
      
      # Append the new data
      data["rewards_list"].append(joint_estimates_list)
      data["ground_truth"].append(ground_truth_list)

      # Write the updated data back to the file
      with open(filename, 'w') as file:
          json.dump(data, file, indent=4)

    return joint_estimates
  

  
  def build_estimator(dataset : TFOffpolicyDataset):
    activation_fn = tf.nn.relu
    kernel_initializer = tf.keras.initializers.GlorotUniform()
    hidden_dims = (64, 64)
    input_spec = (dataset.spec.observation, dataset.spec.action)
    nu_network = ValueNetwork(
        input_spec,
        fc_layer_params=hidden_dims,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        last_kernel_initializer=kernel_initializer)
    output_activation_fn = tf.math.square if zeta_pos else tf.identity
    zeta_network = ValueNetwork(
        input_spec,
        fc_layer_params=hidden_dims,
        activation_fn=activation_fn,
        output_activation_fn=output_activation_fn,
        kernel_initializer=kernel_initializer,
        last_kernel_initializer=kernel_initializer)

    nu_optimizer = tf.keras.optimizers.Adam(nu_learning_rate, clipvalue=1.0)
    zeta_optimizer = tf.keras.optimizers.Adam(zeta_learning_rate, clipvalue=1.0)
    lam_optimizer = tf.keras.optimizers.Adam(nu_learning_rate, clipvalue=1.0)

    estimator = NeuralDice(
        dataset.spec,
        nu_network,
        zeta_network,
        nu_optimizer,
        zeta_optimizer,
        lam_optimizer,
        gamma,
        zero_reward=zero_reward,
        f_exponent=f_exponent,
        primal_form=primal_form,
        reward_fn=reward_fn,
        primal_regularizer=primal_regularizer,
        dual_regularizer=dual_regularizer,
        norm_regularizer=norm_regularizer,
        nu_regularizer=nu_regularizer,
        zeta_regularizer=zeta_regularizer)
  
    return estimator  



  run_cross_fitting()
  
  


if __name__ == '__main__':
  app.run(main)
