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

    scores = [x['results']['fid50k_full'] for x in json_list]
    kimg = [int(re.split('\W+', x['snapshot_pkl'])[-2]) for x in json_list]

    return scores, kimg


def plot_scores(paths, labels, title):

    kimg_list = []
    scores_list = []
    for path in paths:
        scores, kimg = read_json(path)
        kimg_list.append(kimg)
        scores_list.append(scores)


    for kimg, scores, label in zip(kimg_list, scores_list, labels):
        plt.plot(kimg, scores, label=label)

    plt.xlabel('kimg')
    plt.ylabel('FID')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.ylim(0, 20)
    plt.savefig(f'test.png')



paths = ['/scratch/ssd002/home/mengyu/stylegan2-ada-pytorch/stylegan2ada/condnopac/00000-NABirds-cond-auto4/metric-fid50k_full.jsonl',
         '/scratch/ssd002/home/mengyu/metric-fid50k_full.jsonl'
         ]

labels = ['Conditional Non-Pac',
          'Conditional 2-Pac']

plot_scores(paths, labels, 'Conditional Non-Pac vs Pac')
