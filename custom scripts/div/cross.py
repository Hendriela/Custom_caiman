#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 30/07/2022 13:03
@author: hheise

"""
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt


# A custom function to calculate
# probability distribution function
def pdf(x):
    mean = np.mean(x)
    std = np.std(x)
    y_out = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(- (x - mean) ** 2 / (2 * std ** 2))
    return y_out


# To generate an array of x-values
x = np.arange(-2, 2, 0.1)

# To generate an array of
# y-values using corresponding x-values
y = pdf(x)

y_shift = np.roll(y, 10)

# Plotting the bell-shaped curve
plt.style.use('seaborn')
plt.figure(figsize=(6, 6))
plt.plot(x, y)
plt.plot(x, y_shift)
plt.show()

crosscorr = np.correlate(y, y_shift, mode='full')
plt.figure()
plt.plot(np.arange(-(len(y)-1), len(y)), crosscorr)

crosscorr1 = np.correlate(y_shift, y, mode='full')
plt.plot(np.arange(-(len(y)-1), len(y)), crosscorr1)
