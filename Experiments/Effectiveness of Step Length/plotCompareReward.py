import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def main():
    # rewardList1 = reloadReward("D:/pythonWork/indoor Location/forth-code/saveModel/reward-130_NN100.mat")
    # rewardList2 = reloadReward("D:/pythonWork/indoor Location/forth-code/saveModel/reward-150_NN100.mat")
    # rewardList3 = reloadReward("D:/pythonWork/indoor Location/forth-code/saveModel/reward-180_NN100.mat")
    # rewardList4 = reloadReward("D:/pythonWork/indoor Location/forth-code/saveModel/reward-200_NN100.mat")
    # rewardList5 = reloadReward("D:/pythonWork/indoor Location/forth-code/saveModel/reward-200_NN200.mat")

    rewardList1 = reloadReward("D:/pythonWork/indoor Location/forth-code/saveModel/MeetingData/reward-50_NN30.mat")
    rewardList2 = reloadReward("D:/pythonWork/indoor Location/forth-code/saveModel/MeetingData/reward-60_NN50.mat")
    rewardList3 = reloadReward("D:/pythonWork/indoor Location/forth-code/saveModel/MeetingData/reward-80_NN50.mat")
    rewardList4 = reloadReward("D:/pythonWork/indoor Location/forth-code/saveModel/MeetingData/reward-90_NN50.mat")
    rewardList5 = reloadReward("D:/pythonWork/indoor Location/forth-code/saveModel/MeetingData/reward-100_NN50.mat")

    figure, ax = plt.subplots()
    # plt.plot(np.arange(len(rewardList1)), SwapValue(rewardList1), label = 'step-130')
    # plt.plot(np.arange(len(rewardList2)), SwapValue(rewardList2), label = 'step-150')
    # plt.plot(np.arange(len(rewardList3)), SwapValue(rewardList3), label = 'step-180')
    # plt.plot(np.arange(len(rewardList4)), SwapValue(rewardList4), label = 'step-200')
    # plt.plot(np.arange(len(rewardList5)), SwapValue(rewardList5), label = 'step-200-Best')

    plt.plot(np.arange(len(rewardList1)), SwapValue(rewardList1), label = 'step-50')
    plt.plot(np.arange(len(rewardList2)), SwapValue(rewardList2), label = 'step-60')
    plt.plot(np.arange(len(rewardList3)), SwapValue(rewardList3), label = 'step-80')
    plt.plot(np.arange(len(rewardList4)), SwapValue(rewardList4), label = 'step-90')
    plt.plot(np.arange(len(rewardList5)), SwapValue(rewardList5), label = 'step-100-Best')

    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
    plt.xlabel('episode',font2)
    plt.ylabel('Total moving reward',font2)
    plt.grid(color="grey", linestyle=':', linewidth=0.5)
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.legend(prop={'family': 'Times New Roman', 'size': 12}, loc='lower right')
    # plt.savefig('Meet_Reward.png', bbox_inches='tight', dpi=500)
    plt.show(dpi=500)

def reloadReward(filePath):
    reward = loadmat(filePath)
    rewardList = reward['array'][0]
    return rewardList

def SwapValue(x):         #归一化
    max = np.max(x)
    min = np.min(x)
    k = (1-0) / (max-min)
    value = k *(x-min)+0
    return value

if __name__ == '__main__':
    main()
    pass