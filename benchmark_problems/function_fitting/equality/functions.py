import numpy as np


def y1_f_function(x1, frequency=5):
    return 2*np.sin(frequency * x1)

def y2_f_function(x1, frequency=5):
    return - (np.sin(frequency * x1)**2 + x1**2)

# def y1_f_function(x1, frequency=5):
#     return np.sin(frequency * x1)

# def y2_f_function(x1, frequency=5):
#     return np.cos(frequency * x1)

# def y1_f_function(x1):
#     return np.sin(x1)

# def y2_f_function(x1):
#     return (np.sin(x1)**2)**(1/3)

# def y1_f_function(x1, frequency=5):
#     return 8*x1**3 + 5

# def y2_f_function(x1, frequency=5):
#     return 2*x1 - 1


def get_functions():
    return y1_f_function, y2_f_function
