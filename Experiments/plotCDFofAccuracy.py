import statsmodels.api as sm
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def read(path):
    sample = loadmat(path)
    sample = np.array(sample['array']).flatten()
    ecdf = sm.distributions.ECDF(sample)
    x = np.linspace(min(sample), max(sample))
    y = ecdf(x)
    return x, y

def main():

    "-----Lab------"
    # x1, y1 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/Predict-Lab-Error.mat')
    # x2, y2 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/compare/CompareLib/MLP-Lab-Error.mat')
    # x3, y3 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/compare/CompareLib/Forest-Lab-Error.mat')
    # x4, y4 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/compare/CompareLib/TNSE-Lab-Error.mat')
    # x5, y5 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/compare/CompareLib/TMC-Lab-Error.mat')
    #
    # figure, ax = plt.subplots()
    # plt.step(x1, y1, color='r', marker='o', label='ILCF')
    # plt.step(x2, y2, color='b', marker='v', label='FL-MLP')
    # plt.step(x3, y3, color='green', marker='x', label='ILM-CFBCS')
    # plt.step(x4, y4, color='c', marker='p', label='GA-IPP')
    # plt.step(x5, y5, color='orange', marker='+', label='A3C-IPP')

    # "-----Meeting room------"
    # x1, y1 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/Predict-Meet-Error.mat')
    # x2, y2 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/compare/MLP-Meet-Error.mat')
    # x3, y3 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/compare/Forest-Meet-Error.mat')
    # x4, y4 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/compare/TNSE-Meet-Error.mat')
    # x5, y5 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/compare/TMC-Meet-Error.mat')
    #
    # figure, ax = plt.subplots()
    # plt.step(x1, y1, color = 'r', marker ='o', label='ILCF')
    # plt.step(x2, y2, color='b', marker='v', label='FL-MLP')
    # plt.step(x3, y3, color='green', marker='x', label='ILM-CFBCS')
    # plt.step(x4, y4, color='c', marker='p', label='GA-IPP')
    # plt.step(x5, y5, color='orange', marker='+', label='A3C-IPP')

    # "-----Predict and Original database------"
    # x1, y1 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/Predict-Lab-Error.mat')
    # x2, y2 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/Original-Lab-Error.mat')
    # x3, y3 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/Predict-Meet-Error.mat')
    # x4, y4 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/Original-Meet-Error.mat')
    #
    # figure, ax = plt.subplots()
    # plt.step(x1, y1, color='r', marker='o', label='场景一-预测指纹库')
    # plt.step(x2, y2, color='b', marker='v', label='场景一-原始指纹库')
    # plt.step(x3, y3, color='green', marker='x', label='场景二-预测指纹库')
    # plt.step(x4, y4, color='c', marker='p', label='场景二-原始指纹库')

    # "-----Compare performance of imitation learning------"
    # x1, y1 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/Predict-Lab-Error.mat')
    # x2, y2 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/WithoutImitation-Lab-Error.mat')
    # x3, y3 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/Predict-Meet-Error.mat')
    # x4, y4 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/WithoutImitation-Meet-Error.mat')
    #
    # figure, ax = plt.subplots()
    # plt.step(x1, y1, color='r', marker='o', label='场景一-模仿学习')
    # plt.step(x2, y2, color='b', marker='v', label='场景一-无模仿学习')
    # plt.step(x3, y3, color='green', marker='x', label='场景二-模仿学习')
    # plt.step(x4, y4, color='c', marker='p', label='场景二-无模仿学习')

    # "-----Compare different step length for Lab------"
    # x1, y1 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/Predict-Lab-Error.mat')
    # x2, y2 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/Predict180-Lab-Error.mat')
    # x3, y3 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/Predict150-Lab-Error.mat')
    # x4, y4 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/Predict130-Lab-Error.mat')
    #
    # figure, ax = plt.subplots()
    # plt.step(x1, y1, color='r', marker='o', label='步长-200')
    # plt.step(x2, y2, color='b', marker='v', label='步长-180')
    # plt.step(x3, y3, color='green', marker='x', label='步长-150')
    # plt.step(x4, y4, color='c', marker='p', label='步长-130')

    # "-----Compare different step length for Meeting Room------"
    x1, y1 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/Predict-Meet-Error.mat')
    x2, y2 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/Predict90-Meet-Error.mat')
    x3, y3 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/Predict80-Meet-Error.mat')
    x4, y4 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/Predict60-Meet-Error.mat')
    x5, y5 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fifth code/Predict50-Meet-Error.mat')

    figure, ax = plt.subplots()
    plt.step(x1, y1, color='r', marker='o', label='步长-100')
    plt.step(x2, y2, color='b', marker='v', label='步长-90')
    plt.step(x3, y3, color='green', marker='x', label='步长-80')
    plt.step(x4, y4, color='c', marker='p', label='步长-60')
    plt.step(x5, y5, color='orange', marker='+', label='步长-50')

    # font_set = FontProperties(fname=r"/System/Library/Fonts/STHeiti Medium.ttc", size=15)
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams.update({'font.size': 15})
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
    plt.xlabel('平均定位误差 (m)', size=15)
    plt.ylabel('累积分布', size=15)
    plt.grid(color="grey", linestyle=':', linewidth=0.5)
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.legend( loc = 'lower right')
    # plt.rcParams.update({'font.size': 15})
    plt.tight_layout()
    # plt.savefig('tteettet.pdf', bbox_inches = 'tight') #
    plt.show()

if __name__ == '__main__':
    main()
    pass