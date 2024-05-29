import numpy as np
import datetime
import config
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns 
from scipy.stats import t
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Serif",
    "font.size": 12})
TIME = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")


def accrej_plot(accs_list, s_names_list, save = False, data_str = None):
    rand = accs_list[0][:, 3, :]

    portion_vals = np.linspace(0, 1, 50, endpoint=False)
    n = accs_list[0].shape[0]
    t_crit = t.ppf(0.975, n - 1)  
    u_names = ['tu', 'eu', 'au']

    for i in range(accs_list[0].shape[1]-1):
        # score_mean  = accs[:, i, :].mean(axis=0)
        # score_err = accs[:, i, :].std(axis=0) * t_crit / np.sqrt(n)
        # score_mean_base  = accs_base[:, i, :].mean(axis=0)
        # score_err_base = accs_base[:, i, :].std(axis=0) * t_crit / np.sqrt(n)

        for j, accs in enumerate(accs_list):
            score_mean  = accs[:, i, :].mean(axis=0)
            score_err = accs[:, i, :].std(axis=0) * t_crit / np.sqrt(n)
            plt.plot(portion_vals * 100, score_mean, label=f"${s_names_list[j][i]}$", linestyle="-")
            plt.fill_between(portion_vals * 100, score_mean - score_err,
                            score_mean + score_err, alpha=0.2)
        # plt.plot(portion_vals * 100, score_mean_base, label=f"${s_names_base[i]}$", linestyle="--")
        # plt.fill_between(portion_vals * 100, score_mean_base - score_err_base,
        #                 score_mean_base + score_err_base, alpha=0.2)
        plt.plot(portion_vals * 100, rand.mean(axis=0), label="random", color="black", linestyle="--")
        plt.fill_between(portion_vals*100, rand.mean(axis=0) - rand.std(axis=0)* t_crit / np.sqrt(n), 
                         rand.mean(axis=0) + rand.std(axis=0)* t_crit / np.sqrt(n), alpha=0.2, color="black")
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        plt.xlabel("Rejection in \%")
        plt.ylabel("Accuracy in \%")
        plt.legend()
        # plt.title("TVD")
        if save:
            plt.savefig(f"./results/accr_{u_names[i]}_{data_str}.pdf")
        plt.show()
    

def uncertainty_hist(tu_correct, tu_incorrect, method):
    plt.hist(tu_correct, bins=20, alpha=0.5, label="Correct")
    plt.hist(tu_incorrect, bins=20, alpha=0.5, label="Incorrect")
    plt.xlabel(f"Total uncertainty ({method})")
    plt.ylabel("Frequency")
    plt.legend()
    if config.SAVE:
        plt.savefig(f"./results/uncertainty_{config.DATA}_{config.RUNS}_{config.NUM_MEMBERS}_{method}_{TIME}.pdf")
    plt.show()

def hist(df, bins = 11, score='TU_var', save = False, data_str = None):
  custom_palette = {True:"darkolivegreen" , False: "maroon"}
  sns.histplot(data=df, x=score, hue='correct', bins = bins, stat = 'proportion', common_norm = False, alpha = 0.6, palette=custom_palette)
  plt.legend(labels=['Correct', 'Incorrect'])
  score_label = score.split("_")
  if len(score_label) > 1:
        plt.xlabel(f"${score_label[0]}_{{{score_label[1]}}}$")
  else:
        plt.xlabel(f"${score}$")
  if save:
    plt.savefig(f"./results/hist_{score}_{data_str}.pdf")
  plt.show()