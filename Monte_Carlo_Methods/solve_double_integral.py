import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D

def create_sample(num, lower, upper, distribute_type):
    sample_list = []
    if distribute_type == "uniform":
        sample_list = np.random.uniform(lower, upper, num)
    return list(sample_list)


def estimator(run_times, distribute_type, sample_size, lower_x, upper_x, lower_y, upper_y, integral_func):
    mean = 0
    variance = 0
    val_list = []
    
    for i in range(run_times):
        list_x = create_sample(sample_size, lower_x, upper_x, distribute_type)
        list_y = create_sample(sample_size, lower_y ,upper_y, distribute_type)
        sum_val = 0
        if sample_size == 500 and i == 1:
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            #设置标题
            ax1.set_title('The distribution of x and y')
            #设置X轴标签
            plt.xlabel('X')
            #设置Y轴标签
            plt.ylabel('Y')
            #画散点图
            ax1.scatter(list_x,list_y,c = 'r',marker = 'o')
            #设置图标
            plt.legend('x1')
            #显示所画的图
            plt.show()

        for j in range(len(list_x)):
            sum_val += integral_func(list_x[j], list_y[j]) * (upper_y - lower_y) * (upper_x - lower_x)
        val_list.append(sum_val / sample_size)
        
    
    mean = sum(val_list) / len(val_list)
    for val in val_list:
        variance += (mean - val)**2
    variance /= sample_size
    return mean, variance, val_list


def MC(run_times,distribute_type,  sample_size_list, lower_x, upper_x, 
       lower_y, upper_y, integral_func, save_file):
    mean_list = []
    variance_list = []
    
    index_list = [i + 1 for i in np.arange(run_times)]
    index_list += ['mean', 'variance']
    mean_list = []
    variance_list = []

    record_table = pd.DataFrame(
        np.arange((run_times + 2) * len(sample_size_list)).reshape((run_times + 2), len(sample_size_list)), 
                                columns = sample_size_list, 
                               index = index_list)

    val_set = [] # for visualization
    fig, ax = plt.subplots()
    x_list = [i + 1 for i in range(run_times)]
    
    for sample_num in sample_size_list:
        mean, variance, val_list = estimator(run_times, distribute_type, sample_num, 
                                             lower_x, upper_x, lower_y, upper_y, integral_func)
        val_set.append(val_list)
        val_set2 = val_set
        
        ax.plot(x_list, val_list, label = str(sample_num) + "samples")
        ax.set_xlabel("The run times")
        ax.set_ylabel("The estimated integral value")
        ax.set_title("The estimated integral value  ")
        
        val_list.append(mean)
        val_list.append(variance)

        record_table[sample_num] = val_list
    record_table.to_csv(save_file, sep = ',')

##########################

sample_size_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500]
run_times = 100
save_file = "result3-2.csv"

lower_x = 2
upper_x = 4
lower_y = -1
upper_y = 1
distribute_type = "uniform"

def integral_func(x, y):
    return (y**2 * np.exp(- y**2) + x**4 * np.exp(-x**2)) / (x * np.exp(-x**2))

    
MC(run_times = run_times, distribute_type=distribute_type, sample_size_list =sample_size_list,
  lower_x = lower_x, upper_x = upper_x, lower_y = lower_y, upper_y = upper_y, integral_func=integral_func,
  save_file = save_file)
print("------")