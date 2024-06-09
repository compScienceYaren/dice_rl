#!/usr/bin/env python3
import os

# load_dir = "./tests/testdata"
# env_name = "reacher"

# load_dir = "./tests/testdata"
# env_name = "frozenlake"

# load_dir = "./tests/testdata"
# load_dir_gen = "./tests/testdata/CartPole-v0"
# env_name = "cartpole"

load_dir = "./tests/testdata"
env_name = "grid"

def generate(): 

    commands = []
    seed_limit = 20
    num_trajectory = 200
    max_trajectory_length=100

    for seed in range(seed_limit):
        for alpha in [0.0, 1.0]:
            commands.append(f"python3 scripts/create_dataset.py --save_dir=./tests/testdata --load_dir={load_dir} --env_name={env_name} --num_trajectory={num_trajectory} --max_trajectory_length={max_trajectory_length} --alpha={alpha} --seed={seed} --tabular_obs=0")

    for command in commands:
        os.system(command)


def run_split_and_basic_dice():

    commands_split2 = []
    commands_split5 = []
    commands_original = []
    seed_limit = 20
    num_trajectory = 200
    max_trajectory_length=100
    training_steps=10000

    # SplitDICE (sample splitting with 2-fold cross-fitting)

    for seed in range(seed_limit):
        commands_split2.append(f"python3 scripts/run_neural_dice_split.py --save_dir=./tests/testdata --load_dir=./tests/testdata --env_name={env_name} --num_trajectory={num_trajectory} --max_trajectory_length={max_trajectory_length} --alpha=0.0 --seed={seed} --tabular_obs=0 --num_steps={training_steps} --fold_number=2")

    for command in commands_split2:
        os.system(command)

    # SplitDICE (sample splitting with 5-fold cross-fitting)

    for seed in range(seed_limit):
        commands_split5.append(f"python3 scripts/run_neural_dice_split.py --save_dir=./tests/testdata --load_dir=./tests/testdata --env_name={env_name} --num_trajectory={num_trajectory} --max_trajectory_length={max_trajectory_length} --alpha=0.0 --seed={seed} --tabular_obs=0 --num_steps={training_steps} --fold_number=5")

    for command in commands_split5:
        os.system(command)

    # Basic DICE

    for seed in range(seed_limit):
        commands_original.append(f"python3 scripts/run_neural_dice.py --save_dir=./tests/testdata --load_dir=./tests/testdata --env_name={env_name} --num_trajectory={num_trajectory} --max_trajectory_length={max_trajectory_length} --alpha=0.0 --seed={seed} --tabular_obs=0 --num_steps={training_steps}")

    for command in commands_original:
        os.system(command)


generate()
# run_split_and_basic_dice()


