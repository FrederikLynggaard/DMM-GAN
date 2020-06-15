import argparse
import os
import json

import matplotlib.pyplot as plt
import numpy as np


METRICS_DICT = {'FID': 'FID', 'IS': 'Inception score', 'R': 'R-precision (%)'}

def parse_args():
    parser = argparse.ArgumentParser(description='Plot IS, FID, and R-precision')
    parser.add_argument('--scores_dir', dest='scores_dir', type=str, default='')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    model_names = os.listdir(args.scores_dir)
    model_names = [x[:x.rfind('_')] for x in model_names]

    plot_info = {'AttnGAN': '--',
                 'MirrorGAN': '--',
                 'DM-GAN': '--',
                 'Baseline': '-',
                 'DMM-STREAM': '-',
                 'DMM-GLDM': '-',
                 'DMM-Full': '-'}

    for metric in ['FID', 'IS', 'R']:
        # prettify graph
        ax = plt.gca()
        ax.xaxis.grid(True)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(METRICS_DICT[metric])

        if metric == 'FID':
            ax.set(ylim=(5, 35))

        for model_name in plot_info.keys():
            with open(os.path.join(args.scores_dir, model_name + '_scores.json')) as json_file:
                data = json.load(json_file)

            if metric == 'FID':
                yvals = [float(data[i][metric]) for i in data.keys()]
            elif metric == 'R':
                yvals = [float(data[i][metric]['mean']) * 100 for i in data.keys()]
            else:
                yvals = [float(data[i][metric]['mean']) for i in data.keys()]

            plt.plot(list(data.keys())[1:], yvals[1:], plot_info[model_name], label=model_name)

        plt.gca().set_prop_cycle(None)

        for model_name in plot_info.keys():
            with open(os.path.join(args.scores_dir, model_name + '_scores.json')) as json_file:
                data = json.load(json_file)

            if metric == 'FID':
                yvals = [float(data[i][metric]) for i in data.keys()]
                x_index = np.argmin(yvals)
            elif metric == 'R':
                yvals = [float(data[i][metric]['mean']) * 100 for i in data.keys()]
                x_index = np.argmax(yvals)
            else:
                yvals = [float(data[i][metric]['mean']) for i in data.keys()]
                x_index = np.argmax(yvals)

            plt.plot(x_index - 1, yvals[x_index], marker='D', markerfacecolor='none', markersize=10, markeredgewidth=2)
        ax.legend()
        plt.show()