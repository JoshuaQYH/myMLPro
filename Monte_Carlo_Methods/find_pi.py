import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

def create_point(point_num):
    """
    @param:
        point_num: the number of the generated points
    
    @return:
        inner_list: the list of points in circle
        outer_list: the list of points out of circle
    @describe:
        creat some points in the square 
    """
    point_list = np.random.uniform(0, 1, size = (point_num, 2))
    inner_list = []
    outer_list = []
    for point in point_list:
        if point[0] * point[0] + point[1] * point[1] > 1:
            outer_list.append(point)
        else:
            inner_list.append(point)
    return inner_list, outer_list

def transform(point_list):
    # To split the point list into a x-list and y-list for visualization
    x_list = []
    y_list = []
    for p in point_list:
        x_list.append(p[0])
        y_list.append(p[1])
    return x_list, y_list


def estimate_pi(point_num, run_times):
    """
    @params:
        point_num: the number of the random sampling points
        run_times:  the sampling times
    @return:
        pi_list: the estimated pi in every run time
        mean: the mean of all estimated pi value
        variance: the variance of all estimated pi value
    @describe:
        repeat "run_times" times to estimate pi with the given "point_num" points 
    """
    pi_list = []
    mean = 0
    variance = 0
    
    for i in range(run_times):
        inner_list, outer_list = create_point(point_num)
        inner_num = len(inner_list)
        outer_num = len(outer_list)
    
        sampling_pi = inner_num / point_num
        pi_list.append(sampling_pi)
        if i == 0:
            in_x, in_y = transform(inner_list)
            out_x, out_y = transform(outer_list)
            
            fig, ax = plt.subplots()
            ax.scatter(in_x, in_y, c = 'r', alpha = 0.5)
            ax.scatter(out_x, out_y, c = 'g',alpha = 0.5)
            ax.set_xlabel("x", fontsize = 15)
            ax.set_ylabel('y', fontsize = 15)
            ax.set_title("The sampling result of " + str(point_num) + " points.")
            
            #plt.show()
    mean = sum(pi_list)/run_times

    for pi in pi_list:
        variance += (mean - pi)**2
    variance /= run_times - 1
    return pi_list, mean, variance


run_times =100

point_num_list = [20, 50, 100, 200, 300, 500, 1000, 5000]
index_list = [i + 1 for i in np.arange(run_times)]
index_list += ['mean', 'variance']
mean_list = []
variance_list = []

record_table = pd.DataFrame(
    np.arange((run_times + 2) * len(point_num_list)).reshape((run_times + 2), 8), 
                            columns = point_num_list, 
                           index = index_list)

fig, ax = plt.subplots()
x_list = [i + 1 for i in range(run_times)]
real_pi_list = [math.pi / 4 for i in range(run_times)]


for point_num in point_num_list:
    pi_list, mean, variance = estimate_pi(point_num, run_times)
    ax.plot(x_list, pi_list, label = str(point_num) + ' points')
    pi_list.append(mean)
    pi_list.append(variance)
    record_table[point_num] = pi_list 
ax.plot(x_list, real_pi_list, c= 'r', label = "pi/4")

ax.legend()
plt.show()
record_table.to_csv("result1.csv", sep = ',')


