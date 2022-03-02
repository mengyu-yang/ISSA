import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import json
import re


def read_json(JSON_PATH):
    with open(JSON_PATH, 'r') as json_file:
        json_list = list(json_file)

    for i in range(len(json_list)):
        json_list[i] = json.loads(json_list[i])

    E_grad = [x['Grad/E_avg_grad']['mean'] for x in json_list]
    synthesis_grad = [x['Grad/G_syn_avg_grad']['mean'] for x in json_list]
    mapping_grad = [x['Grad/G_map_avg_grad']['mean'] for x in json_list]
    kimg = [int(x['Progress/kimg']['mean']) for x in json_list]

    return E_grad, synthesis_grad, mapping_grad, kimg


def plot_scores(path, labels, title):

    E_grad, synthesis_grad, mapping_grad, kimg = read_json(path)

    plt.plot(kimg, E_grad, 'b', label='Encoder')
    # plt.plot(kimg, synthesis_grad, label='Synthesis network')
    plt.plot(kimg, mapping_grad, 'g', label='Mapping network')

    plt.xlabel('kimg')
    plt.ylabel('Avg Gradient Magnitude')
    plt.title(title)
    plt.legend()
    plt.grid()
    # plt.ylim(0, 150)
    plt.savefig(f'test.png')


paths = '/scratch/ssd002/home/mengyu/stylegan2-ada-pytorch/stylegan2ada/celeba_grad_stats/00004-celeba_train-ssize2-encoder-loss_orig-Encoder-cond-auto-cw1.0-zw1.01-ge_lr0.0025-d_lr0.0025-nlayer2-noplmixing/stats.jsonl'


labels = ['Average module gradient magnitudes']

plot_scores(paths, labels, 'Average Module Gradient Magnitudes')
