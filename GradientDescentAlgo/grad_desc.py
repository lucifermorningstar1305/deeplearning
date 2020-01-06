import numpy as np
from numpy import *
import pandas as pd

## y = mx + b

def compute_error_points(b, m, points):
    totalError = 0
    for i in range(0,len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - ( m * x + b )) ** 2
    return totalError / float(len(points))


def runner(points, starting_b, starting_m, num_iterations, learning_rate):
    b = starting_b
    m = starting_m
    for i in range(0, num_iterations):
        b, m = step_gradient(np.array(points), b, m, learning_rate)
    return [b,m]

def step_gradient(points, b_current, m_current, learning_rate):
    b_gradient = 0 
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i,0]
        y = points[i,1]
        b_gradient += -(2 / N) * (y - ( (m_current * x) + b_current))
        m_gradient += -(2 / N) * x * ( y - ( (m_current * x) + b_current )) 
        b_current = b_current - (learning_rate * b_gradient)
        m_current = m_current - (learning_rate * m_gradient)

    
    return [b_current, m_current]





def run():
    points = genfromtxt("data.csv",delimiter = ',')
    learning_rate = 0.001
    m = 0 ## initial value
    b = 0 ## initial value
    print("Starting Gradient Descent Runner .....")
    print("Initial Config \n b = 0.0 \n m = 0.0 \n Error value = {0}".format(compute_error_points(b,m,points)))
    [b, m] = runner(points, b, m, 1000, learning_rate)
    print("After iterations {0}, \n b = {1} \n m = {2} \n Error value = {3}".format(1000, b, m, compute_error_points(b,m,points)))
    

if __name__ == '__main__':
    run()
