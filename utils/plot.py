from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import tensorflow.compat.v2 as tf
from pathlib import Path
import os
import json
import scipy.stats as stats
from scipy.stats import f
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd
import seaborn as sns
import numpy as np
from typing import List, Optional
tf.compat.v1.enable_v2_behavior()

# def process_tensor_events(event_acc, variable_name, get_steps):

#     # Get tensor events for the specified variable
#     tensor_events = event_acc.Tensors(variable_name)
    
#     # Process the tensor events
#     data = []
#     for tensor_event in tensor_events:
#         w = tensor_event.wall_time
#         s = tensor_event.step
#         t = tf.make_ndarray(tensor_event.tensor_proto)
#         data.append((w, s, t))

#     # steps = [entry[1] for entry in data]
#     # values = [entry[2].item() for entry in data]

#     if get_steps:
#         return [entry[1] for entry in data]
#     else:
#         return [entry[2].item() for entry in data]

# def makePlot(file_name):

#     event_acc = EventAccumulator(file_name, size_guidance={"scalars": 0})
#     event_acc.Reload()

#     steps = process_tensor_events(event_acc, 'nu_zero', get_steps=True)
    
#     variables = ['nu_zero', 'lam', 'dual_step', 'constraint', 'nu_reg', 'zeta_reg', 'lagrangian', 'overall']
#     variable_dict = {}
#     # Iterate over each variable and process its tensor events
#     for var in variables:
#         variable_dict[var] = process_tensor_events(event_acc, var, get_steps=False)
    
#     for var, values in variable_dict.items():
#         print(f"{var}: {values}")
    
#     print(f"{'steps'}: {steps}")

#     # Plot the values
#     plt.figure(figsize=(10, 6))

#     for var, val in variable_dict.items():
#         plt.plot(steps, val, label=var)

#     plt.xlabel('Step')
#     plt.ylabel('Value')
#     plt.title('Summary Scalars over Steps')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('./plots/plot.png')


def get_mse_values():

    _, rewards_list2, ground_truth = load_reward_avg(filename= "./results/frozenlake/log_fold2.json")
    _, rewards_list5, _ = load_reward_avg(filename= "./results/frozenlake/log_fold5.json")
    _, rewards_original = load_original(filename= "./results/frozenlake/log_original.json")

    rewards_array2 = np.array(rewards_list2)
    rewards_array5 = np.array(rewards_list5)
    rewards_array_og = np.array(rewards_original)

    estimates2 = []
    for rewards_list in rewards_array2:
        estimates2.append(rewards_list[-1])

    estimates5 = []
    for rewards_list in rewards_array5:
        estimates5.append(rewards_list[-1])

    estimates_og = []
    for rewards_list in rewards_array_og:
        estimates_og.append(rewards_list[-1])

    return np.mean((ground_truth - estimates_og) ** 2), np.mean((ground_truth - estimates2) ** 2), np.mean((ground_truth - estimates5) ** 2)


def calculate_variance_bias_and_std():

    _, rewards_list2, ground_truth = load_reward_avg(filename= "./results/frozenlake/log_fold2.json")
    _, rewards_list5, _ = load_reward_avg(filename= "./results/frozenlake/log_fold5.json")
    _, rewards_original = load_original(filename= "./results/frozenlake/log_original.json")

    rewards_array2 = np.array(rewards_list2)
    rewards_array5 = np.array(rewards_list5)
    rewards_array_og = np.array(rewards_original)

    log_data = []

    # 2-fold

    estimates2 = []
    for rewards_list in rewards_array2:
        estimates2.append(rewards_list[-1])
    variance_fold2 = np.var(estimates2)
    bias_fold2 = abs(np.mean(ground_truth - estimates2))
    std_dev_fold2 = np.std(estimates2)

    info_dict_fold2 = {
        "mse": np.mean((ground_truth - estimates2) ** 2),
        "variance": variance_fold2.tolist(),
        "bias": bias_fold2.tolist(),
        "std_dev": std_dev_fold2.tolist()
    }

    log_data.append({
        'scenario': "2-fold",
        'metrics': info_dict_fold2
    })

    # 5-fold

    estimates5 = []
    for rewards_list in rewards_array5:
        estimates5.append(rewards_list[-1])
    variance_fold5 = np.var(estimates5)
    bias_fold5 = abs(np.mean(ground_truth - estimates5))
    std_dev_fold5 = np.std(estimates5)

    info_dict_fold5 = {
        "mse": np.mean((ground_truth - estimates5) ** 2),
        "variance": variance_fold5.tolist(),
        "bias": bias_fold5.tolist(),
        "std_dev": std_dev_fold5.tolist()
    }

    log_data.append({
        'scenario': "5-fold",
        'metrics': info_dict_fold5
    })

    # original

    estimates_og = []
    for rewards_list in rewards_array_og:
        estimates_og.append(rewards_list[-1])
    variance_og = np.var(estimates_og)
    bias_og = abs(np.mean(ground_truth - estimates_og))
    std_dev_og = np.std(estimates_og)
    
    info_dict_og = {
        "mse": np.mean((ground_truth - estimates_og) ** 2),
        "variance": variance_og.tolist(),
        "bias": bias_og.tolist(),
        "std_dev": std_dev_og.tolist()
    }

    log_data.append({
        'scenario': "naive",
        'metrics': info_dict_og
    })

    with open("./results/frozenlake/metrics.json", 'w') as file:
        json.dump(log_data, file, indent=4)



def load_reward_avg(filename: str = "./results/log_fold2.json"):
    with open(filename, 'r') as file:
        data = json.load(file)
    steps = np.array(data["steps"])
    rewards_list = [np.array(rewards) for rewards in data["rewards_list"]]
    ground_truth = np.array(data["ground_truth"])
    return steps, rewards_list, ground_truth

def load_original(filename: str = "./results/log_original.json"):
    with open(filename, 'r') as file:
        data = json.load(file)
    steps = np.array(data["steps"])
    rewards_list = [np.array(rewards) for rewards in data["rewards_list"]]
    return steps, rewards_list


def plot_per_step_aggregated(folder: str = "./plots") -> None:
  
    steps, rewards_list, ground_truth = load_reward_avg(filename= "./results/frozenlake/log_fold2.json")
    # steps, rewards_list, ground_truth = load_reward_avg(filename= "./results/frozenlake/log_fold5.json")

    if not os.path.exists(folder):
        os.makedirs(folder)

    base_filename = 'overall_fold2'
    file_extension = ".png"
    filename = os.path.join(folder, base_filename + file_extension)
    counter = 1
    while os.path.exists(filename):
        filename = os.path.join(folder, f"{base_filename}_{counter}{file_extension}")
        counter += 1

    plt.figure(figsize=(10, 6))

    # Convert the list of rewards to a numpy array for easier manipulation
    rewards_array = np.array(rewards_list)
    
    # Calculate the median, 25th percentile, and 75th percentile
    median_rewards = np.median(rewards_array, axis=0)
    percentile_25 = np.percentile(rewards_array, 25, axis=0)
    percentile_75 = np.percentile(rewards_array, 75, axis=0)

    # # Plot the 25th to 75th percentile fill
    # plt.fill_between(steps, percentile_25, percentile_75, color=(0.7, 0.8, 1.0), alpha=0.3)
    # # Plot the median line
    # plt.plot(steps, median_rewards, color='blue')

    # Plot the 25th to 75th percentile fill
    plt.fill_between(steps, percentile_25, percentile_75, color=(1.0, 0.7, 0.4), alpha=0.3)
    # Plot the median line 
    plt.plot(steps, median_rewards, color='#FF4500')

    plt.ylim(0.00, 0.08)
    plt.axhline(y=np.mean(ground_truth), color='r', linestyle='--', label='True value')
    plt.xlabel('Training Steps')
    plt.ylabel('Average per-step reward')
    plt.title(f'SplitDICE with 2-fold Cross-fitting')
    # plt.title(f'SplitDICE with 5-fold Cross-fitting')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)


def plot_with_original(folder: str = "./plots"):

    steps, rewards_split, ground_truth = load_reward_avg(filename= "./results/frozenlake/log_fold5.json")
    _, rewards_original = load_original(filename= "./results/frozenlake/log_original.json")

    if not os.path.exists(folder):
        os.makedirs(folder)

    base_filename = 'original_avg'
    file_extension = ".png"
    filename = os.path.join(folder, base_filename + file_extension)
    counter = 1
    while os.path.exists(filename):
        filename = os.path.join(folder, f"{base_filename}_{counter}{file_extension}")
        counter += 1

    plt.figure(figsize=(10, 6))

    # splitDICE plotting

    # rewards_array_split = np.array(rewards_split)

    # median_rewards = np.median(rewards_array_split, axis=0)
    # percentile_25 = np.percentile(rewards_array_split, 25, axis=0)
    # percentile_75 = np.percentile(rewards_array_split, 75, axis=0)
    # variance_split = np.var(rewards_array_split, axis=0)

    plt.axhline(y=np.mean(ground_truth), color='r', linestyle='--', label='True value')
    # plt.fill_between(steps, percentile_25, percentile_75, color=(0.7, 0.8, 1.0), alpha=0.3)
    # plt.plot(steps, median_rewards, color='blue', label="SplitDICE")

    # originalDICE plotting

    rewards_array_og = np.array(rewards_original)
    
    median_rewards = np.median(rewards_array_og, axis=0)
    percentile_25 = np.percentile(rewards_array_og, 25, axis=0)
    percentile_75 = np.percentile(rewards_array_og, 75, axis=0)

    plt.fill_between(steps, percentile_25, percentile_75, color=(0.6, 0.9, 0.6), alpha=0.3)
    plt.plot(steps, median_rewards, color='green')

    # set the axes

    plt.ylim(0.00, 0.08)
    plt.xlabel('Training Steps')
    plt.ylabel('Average per-step reward')
    plt.title(f'Naive DICE')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)


def plot_mse(folder: str = "./plots"):
    
    def absolute_error(true_value, estimated_value):
        return abs(true_value - estimated_value)
    
    def percentage_mse(mse_split, mse_original):
        split_better_count = 0
        original_better_count = 0

        for mse_s, mse_o in zip(mse_split, mse_original):
            if mse_s < mse_o:
                split_better_count += 1
            elif mse_o < mse_s:
                original_better_count += 1

        total_comparisons = len(mse_split)
        split_better_percentage = int((split_better_count / total_comparisons) * 100)
        original_better_percentage = int((original_better_count / total_comparisons) * 100)
    
        return split_better_percentage, original_better_percentage
    
    _, rewards_fold2, ground_truth = load_reward_avg(filename= "./results/frozenlake/log_fold2.json")
    _, rewards_fold5, _ = load_reward_avg(filename= "./results/frozenlake/log_fold5.json")
    _, rewards_original = load_original(filename= "./results/frozenlake/log_original.json")

    if not os.path.exists(folder):
        os.makedirs(folder)

    base_filename = 'error'
    file_extension = ".png"
    filename = os.path.join(folder, base_filename + file_extension)
    counter = 1
    while os.path.exists(filename):
        filename = os.path.join(folder, f"{base_filename}_{counter}{file_extension}")
        counter += 1

    seed_list = []

    mse_split_fold2 = []
    for i, current_run in enumerate(rewards_fold2):
        # mse_split_fold2.append(mean_squared_error(ground_truth[i], current_run[-1]))
        mse_split_fold2.append(absolute_error(ground_truth[i], current_run[-1]))
        seed_list.append(i)

    mse_split_fold5 = []
    for i, current_run in enumerate(rewards_fold5):
        # mse_split_fold5.append(mean_squared_error(ground_truth[i], current_run[-1]))
        mse_split_fold5.append(absolute_error(ground_truth[i], current_run[-1]))

    mse_original = []
    for i, current_run in enumerate(rewards_original):
        # mse_original.append(mean_squared_error(ground_truth[i], current_run[-1]))
        mse_original.append(absolute_error(ground_truth[i], current_run[-1]))

    split2_better_percentage, original2_better_percentage = percentage_mse(mse_split_fold2, mse_original)
    split5_better_percentage, original5_better_percentage = percentage_mse(mse_split_fold5, mse_original)
    split_2fold, split_5fold = percentage_mse(mse_split_fold2, mse_split_fold5)

    plt.figure(figsize=(10, 6))

    plt.axhline(y=np.mean(mse_original), color='black', linestyle='--', label='Mean error Naive')
    plt.axhline(y=np.mean(mse_split_fold2), color='#FFA500', linestyle='--', label='Mean error 2-fold')
    plt.axhline(y=np.mean(mse_split_fold5), color='red', linestyle='--', label='Mean error 5-fold')
    
    # Plot MSE for original (black circles)
    plt.scatter(seed_list, mse_original, color='black', marker='o', label='Naive DICE')
    
    # Plot MSE for split (yellow triangles)
    plt.scatter(seed_list, mse_split_fold2, color='#FFA500', marker='^', label='SplitDICE with 2-fold')

    # Plot MSE for split (red squares)
    plt.scatter(seed_list, mse_split_fold5, color='red', marker='s', label='SplitDICE with 5-fold')

    plt.xlabel('Seed Number')
    plt.ylabel('Absolute error')
    plt.xticks(seed_list)
    plt.title(f'2-fold={split2_better_percentage}% vs Naive={original2_better_percentage}%\n 5-fold={split5_better_percentage}% vs Naive={original5_better_percentage}%\n 2-fold={split_2fold}% vs 5-fold={split_5fold}% ')
    plt.legend()
    plt.savefig(filename)
    

def calc_significance():

    def absolute_error(true_value, estimated_value):
        return abs(true_value - estimated_value)

    _, rewards_fold2, ground_truth = load_reward_avg(filename= "./results/frozenlake/log_fold2.json")
    _, rewards_fold5, _ = load_reward_avg(filename= "./results/frozenlake/log_fold5.json")
    _, rewards_original = load_original(filename= "./results/frozenlake/log_original.json")

    # Sample data for three groups np.var
    group_og = [absolute_error(ground_truth[i], current_run[-1]) for i, current_run in enumerate(rewards_original)]
    group_fold2 = [absolute_error(ground_truth[i], current_run[-1]) for i, current_run in enumerate(rewards_fold2)]
    group_fold5 = [absolute_error(ground_truth[i], current_run[-1]) for i, current_run in enumerate(rewards_fold5)]

    # Create a DataFrame for ANOVA
    data = {
        'value': group_og + group_fold2 + group_fold5,
        'group': ['Original'] * len(group_og) + ['Fold2'] * len(group_fold2) + ['Fold5'] * len(group_fold5)
    }
    df = pd.DataFrame(data)

    # Perform one-way ANOVA
    model = ols('value ~ C(group)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    p_value = anova_table["PR(>F)"].iloc[0]
    print(p_value)
    if p_value < 0.05:
        # Perform Tukey's HSD post-hoc test
        tukey = pairwise_tukeyhsd(endog=df['value'], groups=df['group'], alpha=0.05)
        print(tukey)
    else:
        print("No significant differences found with ANOVA.")

def f_statistic():
    mse1, mse2, mse3 = get_mse_values()
    print(mse1)
    print(mse2)
    print(mse3)

    # Sample sizes of the three groups (assumed or known)
    n1, n2, n3 = 20, 20, 20 

    # Calculate F-statistics
    F12 = mse1 / mse2
    F13 = mse1 / mse3
    F23 = mse2 / mse3

    # Degrees of freedom for the variances
    df1 = n1 - 1
    df2 = n2 - 1
    df3 = n3 - 1

    # Significance level
    alpha = 0.05

    # Calculate critical values for the F-distribution
    critical_value_12 = f.ppf(1 - alpha / 2, df1, df2)
    critical_value_13 = f.ppf(1 - alpha / 2, df1, df3)
    critical_value_23 = f.ppf(1 - alpha / 2, df2, df3)

    # Compare F-statistics with critical values
    results = {
        "F12": (F12, critical_value_12),
        "F13": (F13, critical_value_13),
        "F23": (F23, critical_value_23)
    }

    for key, (F, critical_value) in results.items():
        if F > critical_value or 1 / F > critical_value:
            print(f"{key}: Variances are significantly different (reject H0). F = {F}, critical value = {critical_value}")
        else:
            print(f"{key}: Variances are not significantly different (fail to reject H0). F = {F}, critical value = {critical_value}")


def box_plot(folder: str = "./plots"):

    def absolute_error(true_value, estimated_value):
        return abs(true_value - estimated_value)
    
    _, rewards_fold2, ground_truth = load_reward_avg(filename= "./results/frozenlake/log_fold2.json")
    _, rewards_fold5, _ = load_reward_avg(filename= "./results/frozenlake/log_fold5.json")
    _, rewards_original = load_original(filename= "./results/frozenlake/log_original.json")

    if not os.path.exists(folder):
        os.makedirs(folder)

    base_filename = 'boxplot'
    file_extension = ".png"
    filename = os.path.join(folder, base_filename + file_extension)
    counter = 1
    while os.path.exists(filename):
        filename = os.path.join(folder, f"{base_filename}_{counter}{file_extension}")
        counter += 1

    errors_fold2 = []
    for i, current_run in enumerate(rewards_fold2):
        errors_fold2.append(absolute_error(ground_truth[i], current_run[-1]))

    errors_fold5 = []
    for i, current_run in enumerate(rewards_fold5):
        errors_fold5.append(absolute_error(ground_truth[i], current_run[-1]))

    errors_original = []
    for i, current_run in enumerate(rewards_original):
        errors_original.append(absolute_error(ground_truth[i], current_run[-1]))

    # Combine all reward data for boxplot
    all_rewards = [errors_original, errors_fold2, errors_fold5]

    red_palette = ["#ff9999", "#ff6666", "#ff3333"]

    # Create a boxplot
    plt.figure(figsize=(5, 3))
    sns.swarmplot(data=all_rewards, palette=red_palette, size=6, marker='o', linewidth=1.2, edgecolor='black')  
    sns.boxplot(data=all_rewards, width=0.5, palette=red_palette, boxprops=dict(edgecolor='black'), medianprops=dict(color='black', linewidth=2)) 

    for i, errors in enumerate(all_rewards):
        avg_error = np.mean(errors)
        plt.plot([i-0.25, i+0.25], [avg_error, avg_error], color='black', linestyle='--', linewidth=1.5)

    plt.xlabel('Models')
    plt.ylabel('Absolute error')
    plt.xticks(ticks=[0, 1, 2], labels=['Naive DICE', 'SplitDICE with 2-fold', 'SplitDICE with 5-fold'])
    plt.tight_layout()
    plt.savefig(filename)



# plot_per_step_aggregated()
# calculate_variance_bias_and_std()
# plot_with_original()
# plot_mse()
box_plot()
# calc_significance()
# f_statistic()








