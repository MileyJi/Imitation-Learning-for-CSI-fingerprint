import statsmodels.api as sm
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def read(path):
    sample = loadmat(path)
    sample = np.array(sample['array']).flatten()
    ecdf = sm.distributions.ECDF(sample)
    x = np.linspace(min(sample), max(sample))
    y = ecdf(x)
    return x, y

def sampleError(path):
    sample = loadmat(path)
    sample = np.array(sample['array']).flatten()
    return sample

def main():
    figure, ax = plt.subplots()
    '-----Lab-----'
    sample = sampleError(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/Predict-Lab-Error.mat')
    sample2 = sampleError(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/Predict180-Lab-Error.mat')
    sample3 = sampleError(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/Predict150-Lab-Error.mat')
    sample4 = sampleError(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/Predict130-Lab-Error.mat')

    '-----Meeting-----'
    # sample = sampleError(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/Predict-Meet-Error.mat')
    # sample2 = sampleError(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/Predict90-Meet-Error.mat')
    # sample3 = sampleError(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/Predict80-Meet-Error.mat')
    # sample4 = sampleError(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/Predict60-Meet-Error.mat')

    random = np.random.RandomState(10)
    len1 = random.rand(len(sample))
    #
    ax.scatter(len1, sample, s=120, color='r', marker='o', alpha=0.6, label='Budget: 200')
    ax.scatter(len1, sample2, s=120, color='green', marker='x', alpha=0.6, label='Budget: 180')
    ax.scatter(len1, sample3, s=120, color='c', marker='v', alpha=0.6, label='Budget: 150')
    ax.scatter(len1, sample4, s=120, color='b', marker='p', alpha=0.6, label='Budget: 130')

    # ax.scatter(len1, sample, s=120, color = 'r', marker ='o', alpha=0.6, label='Budget: 100')
    # ax.scatter(len1, sample2, s=120, color='green', marker='x', alpha=0.6, label='Budget: 90')
    # ax.scatter(len1, sample3, s=120, color='c', marker='v', alpha=0.6, label='Budget: 80')
    # ax.scatter(len1, sample4, s=120, color='b', marker='p', alpha=0.6, label='Budget: 60')

    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
    plt.xlabel('总奖赏值', size=15)
    plt.ylabel('平均定位误差 (m)', size=15)

    step = np.arange(150, 300, 20)
    ax.set_xticklabels(step)
    plt.grid(color="grey", linestyle=':', linewidth=0.5)
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.legend(prop={'size': 12}, loc = 'upper right')
    plt.tight_layout()
    figure.savefig('testststst.pdf', bbox_inches = 'tight', format='pdf')
    plt.show()

if __name__ == '__main__':
    main()
    pass