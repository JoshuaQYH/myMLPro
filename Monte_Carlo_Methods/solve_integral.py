import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.stats import norm
import math


def create_value_byMH(sample_size, goal_f, a, b):
    sample_list = [0 for i in range(sample_size)]
    sample_list[0] = random.uniform(a, b) 
    iterations = sample_size  # the iterations times for calculate new PR
    sigma = 1             # the state transfer matrix's parameter
    t = 0 
    while t < iterations - 1:
        t += 1
        sample_star = norm.rvs(loc = sample_list[t-1], scale=sigma, size=1, random_state=None)
        alpha = min(1, goal_f(sample_star[0])/goal_f(sample_list[t - 1]))
        
        u = random.uniform(0,1)
        if u < alpha:
            sample_list[t] = sample_star[0]
        else:
            sample_list[t] = sample_list[t - 1]
    
    samples = np.array(sample_list)
    minval = np.min(samples)
    maxval = np.max(samples)
    for i in range(len(sample_list)):
        samples[i] = (sample_list[i] - minval) / (maxval - minval)
    result = (b - a) * samples + a
    sample_list = list(result)
    return sample_list

# create random value list in [0, 1]
def create_variable(num_x, distribute_type, a, b, goal_f = None):
    """
    @param:
        num_x: the number of the generating x
        distribute_type: the type of the distribution
        a, b: the lower or upper bound 
    @return:
        list_x:  the generating x list as the independent variable
    """
    list_x = []
    if distribute_type == "uniform":
        list_x = np.random.uniform(a, b, num_x)
    elif distribute_type == "normal":
        list_x = np.random.normal((a + b)/ 2, 0.1, num_x)
    elif distribute_type == "MH":
        goal_f = goal_func
        list_x = create_value_byMH(num_x, goal_f, a, b)
    return list(list_x) 



# calculate integral value using monte carlo method
def estimator(func, a, b, samples_size, run_times, distribute_type):
    """
    @params:
        func: the estiamtor intergral function
        a: the integral lower bound
        b: the integral upper bound
        samples_size: the number  of x sample
        run_times: the repeated times using monte carlo method
        distribute_type: the distribute type of varible x
    @return:
        val_list: the list of estimated values
        mean: the estimated mean value
        variance: the estimated variance value
    """
    val_list = []
    mean = 0
    variance = 0
    
    for i in range(run_times):
        list_x = create_variable(samples_size, distribute_type, a, b)
        sum_val = 0
        for j in list_x:
            if distribute_type == "uniform":
                sum_val += func(j)
            elif distribute_type == "normal":
                sum_val += func(j) / normal_distribute(0.1, (a + b) /2,  j)
            elif distribute_type == "MH":
                sum_val += func(j) / goal_func(j)
        estimated_val =  sum_val / samples_size
        if distribute_type == "uniform":                                               
            estimated_val *= b - a
            
        val_list.append(estimated_val)
        if i == 0:
            # draw the varible distribution
            num_bins = 100
            plt.hist(list_x, num_bins, density=1, facecolor='red', alpha=0.7)
            plt.xlabel("sample value")
            plt.ylabel("the probablity of sample")
            plt.title("The sampling result of " + str(samples_size) + " samples") 
            plt.show()
            
    mean = sum(val_list) / len(val_list)
    for val in val_list:
        variance += (val - mean)**2
    variance /= len(val_list) 
    return val_list, mean, variance    

######################## main 
def MC_(run_times = 100, distribute_type="uniform", sample_size_list = None, upper_b = 1, 
       lower_a = 0, integral_func =  None, save_file = "result.csv"):
    index_list = [i + 1 for i in np.arange(run_times)]
    index_list += ['mean', 'variance']
    mean_list = []
    variance_list = []

    record_table = pd.DataFrame(
        np.arange((run_times + 2) * len(sample_size_list)).reshape((run_times + 2), len(sample_size_list)), 
                                columns = sample_size_list, 
                               index = index_list)

    val_set = [] # for visualization
    for size in sample_size_list:
        val_list, mean, variance = estimator(integral_func, lower_a, upper_b, size, run_times, distribute_type)
        val_set.append(val_list)
        val_list.append(mean)
        val_list.append(variance)
        record_table[size] = val_list
    record_table.to_csv(save_file, sep = ',')
    
    print(record_table)
    fig, axe = plt.subplots(nrows = 1, ncols=1, figsize=(9,4))
    val_plot = axe.boxplot(val_set, vert=True, patch_artist=True, labels = sample_size_list)
    axe.set_title("The estimated integral value  ")
    axe.yaxis.grid(True)
    axe.set_xlabel('The sampling size')
    axe.set_ylabel('The estimated value')
    plt.grid(True)
    plt.show()
    
    
upper_b = 1
lower_a  = 0
# integral function
def f(x):
    return  x**3

# goal distribution
def goal_func(x):
    y = 4 * x**3
    return y

def normal_distribute(sigma, mu, x):
    return  1 / (np.sqrt(2 * math.pi) * sigma) * np.exp(-1 * (x - mu)**2 / (2 * sigma**2))

run_times = 100
distribute_type = "uniform"
sample_size_list = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,1000]

save_file_name = "result1-1.csv"

MC_(run_times = 100, distribute_type = distribute_type, sample_size_list = sample_size_list, 
   upper_b = upper_b, lower_a = lower_a, integral_func =  f, save_file = save_file_name)
