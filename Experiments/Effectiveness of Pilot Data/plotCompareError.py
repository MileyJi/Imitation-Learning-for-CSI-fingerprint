import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def main():
    figure, ax = plt.subplots()
    stepLength = [ '10%', '20%', '30%', '40%', '50%']
    LabError = [ 3.602911945, 3.770844963, 4.148722168, 3.92524759, 3.814280433]
    LabStd = [ 1.911134429, 1.790430524, 1.801132333, 1.560914034, 1.713434211]
    MeetingError = [ 2.983876835, 3.193750054, 3.116716737, 2.929061882, 2.918277863]
    MeetingStd = [ 1.258760912, 1.630570635, 1.391286016, 1.648088739, 1.585955331]

    plt.errorbar(stepLength, LabError, LabStd, fmt='-o', ecolor='c', color='c', elinewidth=1, capsize=3, label='实验室')
    plt.errorbar(stepLength, MeetingError, MeetingStd, fmt='-p', ecolor='r', color='r', elinewidth=1, capsize=3, label='会议室')
    # fmt:'o' ',' '.' 'x' '+' 'v' '^' '<' '>' 's' 'd' 'p'


    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams.update({'font.size': 15})
    plt.xlabel('初始样本数据占比', size=15)
    plt.ylabel('平均定位误差 (m)', size=15)
    plt.grid(color="grey", linestyle=':', linewidth=0.5)
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.legend(loc='upper right')
    plt.ylim(0, 7)
    plt.tight_layout()
    plt.savefig('teststset.pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()