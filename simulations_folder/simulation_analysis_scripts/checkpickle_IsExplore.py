import matplotlib
matplotlib.use('Agg')
import pickle
import os
import pandas as pd 
import matplotlib.pyplot as plt 
# print(data)
import numpy as np
import os
from scipy import stats
from matplotlib.pyplot import figure
import glob
import numpy as np
#import explorE_delete as ed
#figure(num=None, figsize=(15, 15), dpi=60, facecolor='w', edgecolor='k')

#IPW https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/bar_stacked.html
to_check = '2019-08-08_13:19:56/bbUniform0.1BU0.1DfByTrial.pkl'
to_check = 'sim1-2sims/bb0.1BB0.1Df.pkl'
to_check = '2019-08-09_12:39:47/bbEqualMeansEqualPrior32BB0N32Df.pkl'
to_check = '2019-08-09_12:39:47/bbEqualMeansEqualPrior785BB0N785Df.pkl'
to_check = '2019-08-09_12:49:37-20sims_t1/bbEqualMeansEqualPrior785BB0N785Df.pkl' #10?

SMALL_SIZE = 13
MEDIUM_SIZE = 10
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=8.5)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_hist_and_table(df_for_num_steps, df_for_num_steps_ts, df_for_num_steps_unif, num_steps):
        fig_h, ax_h = plt.subplots()
        proportions_unif = df_for_num_steps_unif['sample_size_1'] / num_steps
        proportions = df_for_num_steps['sample_size_1'] / num_steps
        proportions_ts = df_for_num_steps_ts['sample_size_1'] / num_steps
        
        ax_h.hist(proportions, alpha = 0.5, label = "Epsilon Greedy")
        ax_h.hist(proportions_unif, alpha = 0.5, label = "Uniform Random")
        ax_h.hist(proportions_ts, alpha = 0.5, label = "Thompson Sampling")
        ax_h.legend()
        fig_h.suptitle("Prop > 0.75 or ")
       # rows = ["Areferg"]
       # columns = ["Berger"]
       # cell_text = ["ergerg"]
       # the_table = ax_h.table(cellText=cell_text,
         #             rowLabels=rows,
        #              colLabels=columns,
          #            loc='right')

      #  fig_h.subplots_adjust(left=0.2, wspace=0.4)
        data = np.random.uniform(0, 1, 80).reshape(20, 4)
        mean_ts = np.mean(proportions_ts)
        var_ts = np.var(proportions_ts)
        mean_eg = np.mean(proportions)
        var_eg = np.var(proportions)
        prop_lt_25_eg = np.sum(proportions < 0.25) / len(proportions)
        prop_lt_25_ts = np.sum(proportions_ts < 0.25) / len(proportions_ts)

       # prop_gt_25_lt_5_eg = np.sum(> proportions > 0.25) / len(proportions)
       # prop_gt_25_lt_5_ts = np.sum(> proportions_ts > 0.25) / len(proportions_ts)

        data = [[mean_ts, var_ts, prop_lt_25_ts, 3], [mean_eg, var_eg, prop_lt_25_eg,7]]


        final_data = [['%.3f' % j for j in i] for i in data] #<0.25, 0.25< & <0.5, <0.5 & <0.75, <0.75 & <1.0


        table = ax_h.table(cellText=final_data, colLabels=['Mean', 'Variance', 'prop < 0.25', '0.25 < prop < 0.5'], rowLabels = ["Thompson Sampling", "EG"], loc='bottom', cellLoc='center', colColours=['#FFFFFF', '#F3CC32', '#2769BD', '#DC3735'], bbox=[0.25, -0.5, 0.5, 0.3])

        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.auto_set_column_width((-1, 0, 1, 2, 3))

        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.auto_set_column_width((-1, 0, 1, 2, 3))

        # Adjust layout to make room for the table:
        #ax_h.tick_params(axis='x', pad=20)

        #fig_h.subplots_adjust(left=0.2, bottom=0.5)
        #fig_h.tight_layout()
        fig_h.savefig("histograms/condition_prop_n={}.png".format(num_steps), bbox_inches = 'tight')
        fig_h.clf()


def stacked_bar_plot_with_cutoff(df = None, to_check = None, to_check_unif = None, to_check_ipw = None, n = None, num_sims = None, load_df = True, \
                     title = None, percentile_dict_left = None, \
                     percentile_dict_right = None, bs_prop = 0.0,\
                     ax = None, ax_idx = None, to_check_ts = None):
    if load_df == True:
        with open(to_check, 'rb') as f:
            df = pickle.load(f)
        with open(to_check_unif, 'rb') as f:
            df_unif = pickle.load(f)
        if to_check_ipw != None:
            ipw_t1_list =  np.load(to_check_ipw)
        if to_check_ts != None:
            with open(to_check_ts, 'rb') as t:
                df_ts = pickle.load(t)
            

    #print(data)
    
    
    step_sizes = df['num_steps'].unique()
    size_vars = ["n/2", "n", "2*n", "4*n"]
    t1_list = []
    t1_wald_list = []
    wald_stat_list = []
    wald_pval_list = []
    arm1_mean_list = []
    arm2_mean_list = []
    
    arm1_std_list = []
    arm2_std_list = []
    ratio_mean_list = []
    ratio_std_list = []
    t1_simbased_list = []
    
    t1_list_unif = []
    t1_wald_list_unif = []
    var_list = []
    t1_list_ts = []

    for num_steps in step_sizes:
   
        
        df_for_num_steps = df[df['num_steps'] == num_steps]
        df_for_num_steps_unif = df_unif[df_unif['num_steps'] == num_steps]
        df_for_num_steps_ts = df_ts[df_ts['num_steps'] == num_steps]

       # bins = np.arange(0, 1.01, .025)

        plot_hist_and_table(df_for_num_steps, df_for_num_steps_ts, df_for_num_steps_unif, num_steps)

        pval_for_num_steps = df_for_num_steps['pvalue'].mean()
        num_replications = len(df_for_num_steps)
 
       # print(num_replications)
   #     if use_pval == True:
        num_rejected = np.sum(df_for_num_steps['pvalue'] < .05)
        num_rejected_ts = np.sum(df_for_num_steps_ts['pvalue'] < .05)

        num_rejected_unif = np.sum(df_for_num_steps_unif['pvalue'] < .05)
        var = np.var(df_for_num_steps_unif['pvalue'] < .05)
        
        t1 =num_rejected / num_replications
        t1_ts = num_rejected_ts / num_replications
        t1_unif =num_rejected_unif / num_replications
       
        t1_list_unif.append(t1_unif)
        t1_list_ts.append(t1_ts)
        
        t1_list.append(t1)
        var_list.append(var)
        
    t1_list_ts = np.array(t1_list_ts)
    ind = np.arange(len(step_sizes))
 #   print(ind)
  #  print(step_sizes)
    ax.set_xticks(ind)
    ax.set_xticklabels(step_sizes)
   
    print("var", var_list)
    width = 0.23
    capsize = width*8
    width_total = 2*width
    
   
    t1_list = np.array(t1_list)
    t1_list_unif = np.array(t1_list_unif)
    
    t1_se = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_list*(1-t1_list)/num_sims) #95 CI for Proportion
   
    t1_se_unif = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_list_unif*(1-t1_list_unif)/num_sims)
    print(t1_se_unif) #note that power goes to 1.0 for unif, thus error bars
    #print(t1_se_unif)
    p1 = ax.bar(ind, t1_list, width = width, yerr = t1_se, \
                ecolor='black', capsize=capsize, color = 'blue', edgecolor='black')
    
    p2 = ax.bar(ind-width, t1_list_unif, width = width,\
                yerr = t1_se_unif, ecolor='black', \
                capsize=capsize, color = 'red', \
                edgecolor='black')
    if ax_idx == 2:
       leg1 = ax.legend((p1[0], p2[0]), ('Epsilon Greedy Chi Squared', "Uniform Chi Squared"), bbox_to_anchor=(1.0, 1.6))
    
    #leg2 = ax.legend(loc = 2)
    
       ax.add_artist(leg1)
 #   plt.tight_layout()
   # plt.title(title)
#    if ax_idx == 6 or ax_idx == 7 or ax_idx == 8:
    ax.set_xlabel("number of participants = \n n/2, n, 2*n, 4*n")
    ax.set_ylim(0, 0.3)
    ax.axhline(y=0.05, linestyle='--')


    return [t1_list_unif, t1_list, t1_list_ts] #returns [UR Eps_Greedy, TS], in this case, need to return for each step size, but only plotting for one bs, so save step size by model (4x2)

def get_prop_explore(actions_dir, num_sims, n, epsilon, explore = True):
    """
    get prop in cond 1 for when exploring
    """
    step_sizes = [int(np.ceil(n/2)), int(n), int(np.ceil(2*n)), int(np.ceil(4*n))]
  #  debug = actions_dir.split("/")[0] + "/" + actions_dir.split("/")[1] + "/" + actions_dir.split("/")[2] 
  #  print("checking", debug)
  #  print("exists...?")
  #  print(os.path.isdir(debug))
   
    for num_steps in step_sizes:#loop over step sizes, make a hist for each
        print("num_steps", num_steps)
        
        actions_for_all_sims = glob.glob(actions_dir + "/tbb_actions_{}_*.csv".format(num_steps)) #assumes one step size  tbb_actions_
        #print("len(actions_for_all_sims)", len(actions_for_all_sims))
        prop_list = [] #collect over sims

        for action_file in actions_for_all_sims: #looping over num_sims files
            actions_df = pd.read_csv(action_file, skiprows=1)

            if explore == True:
                actions_df_explore = actions_df[actions_df["IsExploring"] == 1]
            else:
                actions_df_explore = actions_df[actions_df["IsExploring"] == 0]

            actions_df_explore_action1 = np.sum(actions_df_explore["AlgorithmAction"] == 1)

            actions_df_explore_action1_prop = actions_df_explore_action1 / len(actions_df_explore)
            prop_list.append(actions_df_explore_action1_prop)
        #now make hist from prop_list, which is over sims
        print(len(prop_list))
        fig_h, ax_h = plt.subplots()
        ax_h.hist(prop_list)
        
        #ax_h.legend()
        if explore == True:
            fig_h.suptitle("Histogram of Proportion of {} Participants Assigned to Condition 1 In Exploration Across 500 Simulations \n Epsilon = {}".format(num_steps, epsilon))
            fig_h.savefig("histograms/IsExploring/condition_prop_ntotal={}_nexplore={}.png".format(num_steps, int(np.floor(num_steps*epsilon))), bbox_inches = 'tight')
        else:
            fig_h.suptitle("Histogram of Proportion of {} Participants Assigned to Condition 1 In Exploitation Across 500 Simulations \n Epsilon = {}".format(num_steps, epsilon))
            fig_h.savefig("histograms/IsExploiting/condition_prop_ntotal={}_nexploit={}.png".format(num_steps, int(np.floor(num_steps*(1-epsilon)))), bbox_inches = 'tight')
        fig_h.clf()
        
            
    #return np.mean(np.array(prop_optimal_list))


def parse_dir(root, root_cutoffs):
    num_sims = 500
    arm_prob= 0.5
    arm_prob_list = [0.2, 0.5, 0.8]
    es_list = [0.5, 0.3, 0.1]
    n_list = [32, 88, 785]
    epsilon = 0.1
#EpsilonGreedyIsEffect/num_sims=5armProb=0.5/es=0.3epsilon=0.1/
    root_dir = root + "/num_sims={}armProb={}".format(num_sims, arm_prob)
    fig, ax = plt.subplots(1,3)
    #fig.set_size_inches(17.5, 13.5)
    ax = ax.ravel()
    i = 0

    root_ts = "../simulation_saves/NoEffect_fixedbs_RL4RL/num_sims=500armProb=0.5/"#hard code for now
    for n in n_list:
        bs = 1
        es_dir = root_dir + "/N={}epsilon={}/".format(n, epsilon)
        ts_dir = root_ts + "/n={}".format(n)

        to_check = glob.glob(es_dir + "/*Prior*{}*{}Df.pkl".format(bs,n))[0] #Has eg, 34 in 348!!
        assert(len(glob.glob(es_dir + "/*Prior*{}*{}Df.pkl".format(bs,n))) == 1)

        to_check_unif = glob.glob(es_dir + "/*Uniform*{}*{}Df.pkl".format(bs, n))[0]
        assert(len(glob.glob(es_dir + "/*Uniform*{}*{}Df.pkl".format(bs, n))) == 1)

        to_check_ts = glob.glob(ts_dir + "/*Prior*{}*{}Df.pkl".format(bs,n))[0] #Has eg, 34 in 348!!
        assert(len(glob.glob(ts_dir + "/*Prior*{}*{}Df.pkl".format(bs,n))) == 1)
  
        actions_dir_eg = es_dir + "bbUnEqualMeansEqualPriorburn_in_size-batch_size={}-{}".format(bs, bs)
        

        prop_explore = get_prop_explore(actions_dir_eg, num_sims, n, epsilon) #averaged over sims
        prop_exploit = get_prop_explore(actions_dir_eg, num_sims, n, epsilon, False) #averaged over sims

     #   to_check_cutoffs = glob.glob(outcome_dir_cutoffs + "/*Prior*{}*{}Df.pkl".format(bs, es))[0] #Has uniform and TS
      #  assert(len(glob.glob(outcome_dir_cutoffs + "/*Prior*{}*{}Df.pkl".format(bs, es))) == 1)

       

        #title = "Power \n  n = {} and {} sims \n Initial Batch Size {} and Batch Size {} \n Arm Prob. {}".format(n, num_sims, bs, bs, arm_prob)
        #percentile_dict_left, percentile_dict_right = hist_and_cutoffs(to_check = to_check_cutoffs, to_check_unif = to_check_unif,\
         #                                                              n = n, num_sims = num_sims, title = title, plot = False) #Note title not used here per say

       # next_df = stacked_bar_plot_with_cutoff(to_check = to_check,to_check_unif = to_check_unif, to_check_ts = to_check_ts,\
  #                                           n = n, num_sims = num_sims,
 #                                              ax = ax[i], ax_idx = i)
#


      #  ax[i].set_title("n = {}".format(n))
       # i += 1

        #df = pd.DataFrame(next_df, columns = ["n/2","n","2n","4n"])
        #df.index = ["Uniform Random Chi Squared","Epsilon Greedy Chi Squared", "Thompson Sampling Chi Squared"]
        #df.to_csv("Tables/Type1Error_n={}_numsims={}.csv".format(n, num_sims)) 

	   
    #title = "Type 1 Error Rate Across {} Simulations For Epsilon = {}".format(num_sims, epsilon)
            #ax[i].set_title(title, fontsize = 55)
            #i +=1
            #fig.suptitle("Type One Error Rates Across {} Simulations".format(num_sims))
   # fig.suptitle(title)
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    #handles, labels = ax[i-1].get_legend_handles_labels()
    
    #fig.legend(handles, labels, loc='upper right', prop={'size': 50})
        #fig.tight_layout()
   # if not os.path.isdir("plots"):
   #       os.mkdir("plots")
  #  print("saving to ", "plots/{}.png".format(title))
  #  fig.tight_layout()
  #  fig.subplots_adjust(top=.8)

  #  fig.savefig("plots/{}.svg".format(title), bbox_inches = 'tight')
  #  plt.show()
  #  plt.clf()
  #  plt.close()


        
root = "EpsilonGreedyNoEffect"
#parse_dir(root, root_cutoffs)
parse_dir(root, root)


