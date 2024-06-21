from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import tensorflow.compat.v2 as tf
from pathlib import Path
import os
import json
from matplotlib.lines import Line2D
import scipy.stats as stats
from scipy.stats import f
from statsmodels.stats.multitest import multipletests
from scipy.stats import kruskal, mannwhitneyu
import statsmodels.api as sm
from scipy.stats import rankdata
from scikit_posthocs import posthoc_dunn
from statsmodels.formula.api import ols
import scikit_posthocs as sp
from statsmodels.stats.diagnostic import lilliefors
from scipy.stats import f_oneway, normaltest, kstest, norm, jarque_bera, anderson, friedmanchisquare, wilcoxon
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd
from scipy.stats import norm, skew, kurtosis
from statsmodels.stats.weightstats import DescrStatsW
import seaborn as sns
import pingouin as pg
import numpy as np
from typing import List, Optional
tf.compat.v1.enable_v2_behavior()

    


def calculate_metrics():

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
    variance_fold2 = np.var(estimates2 / ground_truth)
    bias_fold2 = abs(np.mean(ground_truth - estimates2))
    std_dev_fold2 = np.std(estimates2 / ground_truth)

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
    variance_fold5 = np.var( estimates5 / ground_truth)
    bias_fold5 = abs(np.mean(ground_truth - estimates5))
    std_dev_fold5 = np.std(estimates5 / ground_truth)

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
    variance_og = np.var(estimates_og / ground_truth)
    bias_og = abs(np.mean(ground_truth - estimates_og))
    std_dev_og = np.std(estimates_og / ground_truth)
    
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


    return info_dict_fold2, info_dict_fold5, info_dict_og


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


def plot_split_dice(variable: str = "2-fold", folder: str = "./plots/frozenlake") -> None:

    if variable == "2-fold":
        steps, rewards_list, ground_truth = load_reward_avg(filename= "./results/frozenlake/log_fold2.json")
    if variable == "5-fold":
        steps, rewards_list, ground_truth = load_reward_avg(filename= "./results/frozenlake/log_fold5.json")

    if not os.path.exists(folder):
        os.makedirs(folder)

    base_filename = 'overall_fold2' if variable == "2-fold" else 'overall_fold5' if variable == "5-fold" else None
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

    if variable == "2-fold":
        # Plot the 25th to 75th percentile fill
        plt.fill_between(steps, percentile_25, percentile_75, color=(1.0, 0.7, 0.4), alpha=0.3)
        # Plot the median line 
        plt.plot(steps, median_rewards, color='#FF4500')

        plt.title(f'SplitDICE with 2-fold Cross-fitting', fontsize=18)
    if variable == "5-fold":
        # Plot the 25th to 75th percentile fill
        plt.fill_between(steps, percentile_25, percentile_75, color=(0.7, 0.8, 1.0), alpha=0.3)
        # Plot the median line
        plt.plot(steps, median_rewards, color='blue')

        plt.title(f'SplitDICE with 5-fold Cross-fitting', fontsize=18)

    plt.ylim(0.00, 0.08)
    plt.axhline(y=np.mean(ground_truth), color='r', linestyle='--', label='True value')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Training Steps', fontsize=16)
    plt.ylabel('Average per-step reward', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.savefig(filename)


def plot_naive_dice(folder: str = "./plots"):

    steps, _, ground_truth = load_reward_avg(filename= "./results/frozenlake/log_fold5.json")
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

    rewards_array_og = np.array(rewards_original)
    median_rewards = np.median(rewards_array_og, axis=0)
    percentile_25 = np.percentile(rewards_array_og, 25, axis=0)
    percentile_75 = np.percentile(rewards_array_og, 75, axis=0)

    plt.figure(figsize=(10, 6))
    plt.fill_between(steps, percentile_25, percentile_75, color=(0.6, 0.9, 0.6), alpha=0.3)
    plt.plot(steps, median_rewards, color='green')
    plt.axhline(y=np.mean(ground_truth), color='r', linestyle='--', label='True value')

    # set the axes

    plt.ylim(0.00, 0.08)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Training Steps', fontsize=16)
    plt.ylabel('Average per-step reward', fontsize=16)
    plt.title(f'Naive DICE', fontsize=18)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.savefig(filename)


def plot_error(folder: str = "./plots/frozenlake"):
    
    def relative_error(true_value, estimated_value):
        absolute_err = abs(true_value - estimated_value)
        if true_value != 0:
            return absolute_err / abs(true_value)
    
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

    error_split_fold2 = []
    for i, current_run in enumerate(rewards_fold2):
        error_split_fold2.append(relative_error(ground_truth[i], current_run[-1]))
        seed_list.append(i)

    error_split_fold5 = []
    for i, current_run in enumerate(rewards_fold5):
        error_split_fold5.append(relative_error(ground_truth[i], current_run[-1]))

    error_original = []
    for i, current_run in enumerate(rewards_original):
        error_original.append(relative_error(ground_truth[i], current_run[-1]))

    plt.figure(figsize=(10, 6))

    plt.axhline(y=np.mean(error_original), color='black', linestyle='--', label='Mean error Naive')
    plt.axhline(y=np.mean(error_split_fold2), color='#FFA500', linestyle='--', label='Mean error 2-fold')
    plt.axhline(y=np.mean(error_split_fold5), color='red', linestyle='--', label='Mean error 5-fold')
    
    # Plot MSE for original (black circles)
    plt.scatter(seed_list, error_original, color='black', s = 60, marker='o', label='Naive DICE')
    
    # Plot MSE for split (yellow triangles)
    plt.scatter(seed_list, error_split_fold2, color='#FFA500', s = 60, marker='^', label='2-fold SplitDICE')

    # Plot MSE for split (red squares)
    plt.scatter(seed_list, error_split_fold5, color='red', s = 60, marker='s', label='5-fold SplitDICE')

    plt.xlabel('Seed Number', fontsize=16)
    plt.ylabel('Relative error', fontsize=16)
    plt.xticks(seed_list, fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=14, bbox_to_anchor=(0.5, 1.16), loc='upper center', ncol=3)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0, 3, 1, 4, 2, 5]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=14, bbox_to_anchor=(0.5, 1.16), loc='upper center', ncol=3)
    plt.savefig(filename)
    

def calc_significance():

    def absolute_error(true_value, estimated_value):
        return abs(true_value - estimated_value)
    
    def relative_error(true_value, estimated_value):
        absolute_err = abs(true_value - estimated_value)
        if true_value != 0:
            return absolute_err / abs(true_value)
    
    _, rewards_fold2, ground_truth = load_reward_avg(filename= "./results/frozenlake/log_fold2.json")
    _, rewards_fold5, _ = load_reward_avg(filename= "./results/frozenlake/log_fold5.json")
    _, rewards_original = load_original(filename= "./results/frozenlake/log_original.json")

    # Sample data for three groups 
    group_og = [relative_error(ground_truth[i], current_run[-1]) for i, current_run in enumerate(rewards_original)]
    group_fold2 = [relative_error(ground_truth[i], current_run[-1]) for i, current_run in enumerate(rewards_fold2)]
    group_fold5 = [relative_error(ground_truth[i], current_run[-1]) for i, current_run in enumerate(rewards_fold5)]

    groups = {'Original': group_og, 'Fold2': group_fold2, 'Fold5': group_fold5}
    for group_name, group_data in groups.items():
        print(f"\nAnalyzing group: {group_name}")
        stat, p_value = stats.shapiro(group_data)
        print(f'Shapiro-Wilk Test for {group_name}: W={stat}, p-value={p_value}')

        stat, p_value = kstest(group_data, 'norm', args=(np.mean(group_data), np.std(group_data)))
        print(f'Kolmogorov-Smirnov Test Statistic: {stat}, p-value: {p_value}')

        test_statistic, p_value = lilliefors(group_data)
        print(f'LillieforsTest Statistic: {test_statistic}, p-value: {p_value}')

        result = anderson(group_data)
        print(f'Anderson-Darling Test: Statistic={result.statistic}')
        for i in range(len(result.critical_values)):
            # print(f'Significance level {result.significance_level[i]}: {result.critical_values[i]}')

            if result.significance_level[i] == 5:
                if result.statistic > result.critical_values[i]:
                    print(f'  At the {result.significance_level[i]}% significance level, the null hypothesis is rejected (data is not normally distributed).')
                else:
                    print(f'  At the {result.significance_level[i]}% significance level, the null hypothesis is not rejected (data is normally distributed).')


    # Perform Kruskal-Wallis test
    kruskal_stat, p_value = kruskal(group_og, group_fold2, group_fold5)
    print(f"Kruskal-Wallis test statistic: {kruskal_stat}, p-value: {p_value}")

    if p_value < 0.05:
        print("Significant differences found with Kruskal-Wallis test, performing post-hoc Mann-Whitney U tests:")
        
        # Perform Mann-Whitney U tests for post-hoc analysis
        comparisons = [("Original", group_og, "Fold2", group_fold2),
                       ("Original", group_og, "Fold5", group_fold5),
                       ("Fold2", group_fold2, "Fold5", group_fold5)]
        
        results = []
        for group1_name, group1, group2_name, group2 in comparisons:
            stat, mw_p_value = mannwhitneyu(group1, group2, alternative='two-sided')
            results.append((group1_name, group2_name, mw_p_value))
            print(f"Mann-Whitney U test between {group1_name} and {group2_name}: p-value = {mw_p_value} and stat:{stat}")

        # Adjust p-values for multiple comparisons using the Holm method
        p_values = [result[2] for result in results]
        reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='holm')
        
        print("\nAdjusted p-values and significance after Holm correction:")
        for i, (group1_name, group2_name, _) in enumerate(results):
            print(f"{group1_name} vs {group2_name}: corrected p-value = {pvals_corrected[i]}, significant = {reject[i]}")

    else:
        print("No significant differences found with Kruskal-Wallis test.")




def box_plot(variable: str = "reward", folder: str = "./plots/frozenlake"):
    
    def relative_error(true_value, estimated_value):
        absolute_err = abs(true_value - estimated_value)
        if true_value != 0:
            return absolute_err / abs(true_value)
    
    _, rewards_fold2, ground_truth = load_reward_avg(filename= "./results/frozenlake/log_fold2.json")
    _, rewards_fold5, _ = load_reward_avg(filename= "./results/frozenlake/log_fold5.json")
    _, rewards_original = load_original(filename= "./results/frozenlake/log_original.json")

    if not os.path.exists(folder):
        os.makedirs(folder)


    base_filename = 'boxplot_reward' if variable == "reward" else 'boxplot_error' if variable == "error" else None
    file_extension = ".png"
    filename = os.path.join(folder, base_filename + file_extension)
    counter = 1
    while os.path.exists(filename):
        filename = os.path.join(folder, f"{base_filename}_{counter}{file_extension}")
        counter += 1

    data_fold2 = []
    for i, current_run in enumerate(rewards_fold2):
        if variable == "reward":
            data_fold2.append(current_run[-1])
        if variable == "error":
            data_fold2.append(relative_error(ground_truth[i], current_run[-1]))

    data_fold5 = []
    for i, current_run in enumerate(rewards_fold5):
        if variable == "reward":
            data_fold5.append(current_run[-1])
        if variable == "error":
            data_fold5.append(relative_error(ground_truth[i], current_run[-1]))

    data_original = []
    for i, current_run in enumerate(rewards_original):
        if variable == "reward":
            data_original.append(current_run[-1])
        if variable == "error":
            data_original.append(relative_error(ground_truth[i], current_run[-1]))

    # Combine all reward data for boxplot
    all_data = [data_original, data_fold2, data_fold5]

    red_palette = ["#ff9999", "#ff6666", "#ff3333"]

    # Create a boxplot
    plt.figure(figsize=(10, 6))
    sns.swarmplot(data=all_data, palette=red_palette, size=8, marker='o', linewidth=1.2, edgecolor='black')  
    sns.boxplot(data=all_data, width=0.5, palette=red_palette, boxprops=dict(edgecolor='black'), medianprops=dict(color='black', linewidth=2)) 

    for i, point in enumerate(all_data):
        avg = np.mean(point)
        plt.plot([i-0.25, i+0.25], [avg, avg], color='black', linestyle='--', linewidth=1.5)

    median_line = Line2D([0], [0], color='black', linewidth=2, label='Median')
    mean_line = Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='Mean')
    true_value_line = Line2D([0], [0], color='blue', linestyle=(0, (5, 2, 1, 2)), linewidth=1.5, label='True value')

    
    if variable == "reward":
        plt.legend(handles=[median_line, mean_line, true_value_line], fontsize=14, bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=3)
        plt.axhline(y=np.mean(ground_truth), color='blue', linestyle=(0, (5, 2, 1, 2)), label='True value') 
        plt.ylabel('Average per-step reward', fontsize=16)
    if variable == "error":
        plt.legend(handles=[median_line, mean_line], fontsize=14, bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=2)
        plt.ylabel('Relative error', fontsize=16)

    plt.xlabel('Models', fontsize=16)
    plt.xticks(ticks=[0, 1, 2], labels=['Naive DICE', 'SplitDICE with 2-fold', 'SplitDICE with 5-fold'], fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(filename)



# plot_split_dice(variable="5-fold")
# calculate_metrics()
# plot_naive_dice()
# plot_error()
# box_plot(variable="error")
# calc_significance()







