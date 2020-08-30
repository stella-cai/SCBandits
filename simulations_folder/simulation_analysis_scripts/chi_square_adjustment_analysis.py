import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from glob import glob
import os
from time import sleep
import subprocess

def get_all_file_paths(root_dir):
    to_return = []
    current_level_dfs = glob(f"{root_dir}/*Df.csv")
    if len(current_level_dfs) > 0:
        to_return += [df_path for df_path in current_level_dfs]
    else:
        for subdir in os.listdir(root_dir):
            full_dir = f"{root_dir}/{subdir}" 
            if os.path.isdir(full_dir):
                to_return += get_all_file_paths(full_dir)
    return to_return

if __name__ == "__main__":

    os.chdir("../simulation_scripts")
    os.system("./RunEffectSizeSimsSameArmsTSPPD.sh")
    os.chdir("../simulation_analysis_scripts")

    # wait 10 minutes for all the simulations to finish
    sleep(600)

    save_dir = 'TSPPDNoEffectResampleFast'

    num_sims = 500
    arm_prob = 0.5
    means = {}
    cutoffs = {}
    for n in (32, 88,, 197, 785):
        for c in (0.025, 0.1, 0.2, 0.3, 0.05)
        try:
            file_path = glob(save_dir + f"/num_sims={num_sims}armProb={arm_prob}/N={n}c={c}/*Df.csv")[0]
        except IndexError:
            continue
        df_sims = pd.read_csv(file_path)
        plt.hist(df_sims['stat'])
        plt.title(f"Chi-Square Statistic: n={n}")
        plt.xlabel("Chi-Square Statistic")
        plt.ylabel("# Sims")
        plt.savefig(f'../simulation_analysis_saves/chi_square_cutoff/chi_square_histogram_{n}.png')
        plt.clf()
        cutoff = df_sims['stat'].sort_values().reset_index()['stat'][475]
        print(f"cutoff: {cutoff}")
        cutoffs[f'{n}'] = cutoff
        print(f"chi square mean: {df_sims['stat'].mean()}")
        means[f'{n}'] = df_sims['stat'].mean()

    # delete all the simulation saves
    for save_dir in save_dirs:
        os.system(f"rm -rf {save_dir}")

    # re-run the simulations
    test_scripts = ["RunEffectSizeSimsSameArmsTSPPD.sh", "RunEffectSizeSimsTSPPD.sh"]
    os.chdir("../simulation_scripts")
    for test_script in test_scripts:
        os.system(f"./{test_script}")
    os.chdir("../simulation_analysis_scripts")

    # wait 10 minutes for the new simulations to finish
    sleep(600)

    save_dirs = os.listdir("../simulation_saves")
    save_dirs = ['../simulation_saves/' + d for d in save_dirs]

    get_all_file_paths("../simulation_saves")

    num_sims = 500
    arm_prob = 0.5
    for save_dir in save_dirs:
        for n in (32, 88, 785):
            for file_path in get_all_file_paths("../simulation_saves"):
                print(file_path)
                df_sims = pd.read_csv(file_path)
                cutoff = cutoffs[f'{n}']
                df_positives = df_sims[df_sims['stat'] > cutoff]
                print(f"# above chi-square_cutoff: {len(df_positives)}")
                print(f"% of sims positive: {len(df_positives)/num_sims}")