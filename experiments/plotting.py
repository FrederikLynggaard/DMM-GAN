import argparse
import os
import json

import matplotlib.pyplot as plt


METRICS_DICT = {'FID': 'FID', 'IS': 'Inception score', 'R': 'R-precision (%)'}

def parse_args():
    parser = argparse.ArgumentParser(description='Plot IS, FID, and R-precision')
    parser.add_argument('--scores_dir', dest='scores_dir', type=str, default='')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()


    for metric in ['FID', 'IS', 'R']:
        # prettify graph
        ax = plt.gca()
        ax.xaxis.grid(True)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(METRICS_DICT[metric])

        for file in os.listdir(args.scores_dir):
            with open(os.path.join(args.scores_dir, file)) as json_file:
                data = json.load(json_file)

            if metric == 'FID':
                yvals = [float(data[i][metric]) for i in data.keys()]
            elif metric == 'R':
                yvals = [float(data[i][metric]['mean']) * 100 for i in data.keys()]
            else:
                yvals = [float(data[i][metric]['mean']) for i in data.keys()]

            plt.plot(list(data.keys())[1:], yvals[1:], label=file[:file.rfind('_')])
        ax.legend()
        plt.show()