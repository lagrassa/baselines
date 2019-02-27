import rl_plot.make_plots as mp
SHOW = True


def plot_exp_name(exp_name):
    exp_dict = {}
    algs = ["ppo2", "ddpg"] # naf is so slow to test
    for alg in algs:
        alg_exp_name = alg+exp_name+"success_rates"
        exp_dict[alg] = mp.get_exps_from_root(alg_exp_name, root_dir=".")

    mp.plot_graph(exp_dict, prefix="", title=exp_name, xlab ="number sample batches", ylab="success rate", root_dir="")
    if SHOW:
        import matplotlib.pyplot as plt
        plt.legend()
        plt.savefig("plots/"+exp_name+".png")

if __name__=="__main__":
    import sys
    exp_name = sys.argv[1]
    plot_exp_name(exp_name)
