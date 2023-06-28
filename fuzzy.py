# Project 1 Part 2
# Group 6: Riyadh Ananda, Sam Griffin, Nathan Redd

import pandas as pd 
import numpy as np
import random
import operator
import math

def jaccard_coef(x, y):
    """A function for finding the similarity between two binary vectors"""
    intersection = np.logical_and(x, y)
    union = np.logical_or(x, y)
    difference = intersection.sum() / float(union.sum())
    return difference

# Import excel data

# use abs_path to access the file as long as the excel file is in the same directory
data = pd.read_excel("/content/drive/MyDrive/dataset/Longotor1delta.xls")
# Print the total number of data entries in dataset - should be 5667
#print('Number of Data entries in training set:',len(data))


# Clean the data

# Include all rows, exclude columns 0 (Public ID), column 1 (Gene), and column 2 (Gene Description), this will be used for data manipulation
data = data.iloc[:, 3:6]
#print('Cleaned dataset:', len(data), 'entries - this should still be 5667')
#print(data.head())  # make sure the correct columns were selected


# Normalize the data

# for each column, find the minimum and the maximum
for col in range(3):
    # reset variables
    min = float('inf')
    max = 0
    
    # for each row check to see if it is less than or greater than teh current min and max
    for row in range(len(data)):
        if data.loc[row][col] < min:
            min = data.loc[row][col]
        if data.loc[row][col] > max:
            max = data.loc[row][col]
            
    # for each value in the selected column, recalculate the new value based on min-max
    for row in range(len(data)):
        if (min - max) != 0:
            data.loc[row][col] = (data.loc[row][col] - min) / (max - min)
#print(data.head())

# Setting parameters

# Entries in dataset
n = len(data)
#print(n)
# number of clusters wanted
k = 3
# cluster dimensions
d = 3
# iterations
iterations = 12
# parameter m
m = 2

# Initialize U = [uij] matrix, U^(0)
# referenced geeks-for-geeks for dirichlet implementation
# uij values will be assigned randomly
def UIJMatrix():
  # initialize empty membership matrix
  membership_matrix = []
  # loop through all the elements in dataset - variable n
  for i in range(n):
    # initialize an array to store empty weights
    empty_weights = []
    # initialize sum to zero
    sum=0;
    # nested loop to assign random number to j
    for j in range(k):
      # use np.random built-in function to assign a random integer from 1-10 to variable weight
      weight = np.random.random_integers(1,10)
      # append weight value to empty weight
      empty_weights.append(weight)
      # add variable weight to sum
      sum = sum + weight
    weights = [w/sum for w in empty_weights]
    #append weights to final membership_matrix
    membership_matrix.append(weights)
    #print results
    #print(membership_matrix)
  # use np.randoms dirichlet function to assign to weight variable
  weight = np.random.dirichlet(np.ones(k),n)
  # for formatting, assign weight_array equal to array of weights in order to return and use later
  weight_arr = np.array(weight)
  # return the completed weighted array, this will correspond to [uij]
  return weight_arr

# Centroid Calculation

# Reference Chapter 7 notes
# Psuedo: Cj = ((N-Sigma-i=1)(Uij^m)(xi))/((N-Sigma-i=1)(Uij^m)
def CjCentroids(weight_arr):
  # create an array C, that will store all the centroid values
  Centroids = []
  # for loop to loop through all values
  for i in range(k):
    #weighted sum will account for (N-Sigma-i=1)(Uij^m)
    weighted_sum = np.power(weight_arr[:,i],m).sum()
    cj_calc = []
    # for loop to complete the entire numerator of Cj
    for x in range(d):
      #cj_num accounts for (N-Sigma-i=1)(Uij^m) * Xi
      cj_num = ( data.iloc[:,i].values * np.power(weight_arr[:,i],m)).sum()
      #c_total is the raw calculation of cj, will append to cj_calc and later to centroid for ease of use
      c_total = cj_num/weighted_sum;
      #append unformatted c_total values to cj_calc array
      cj_calc.append(c_total)
    #append cj_calc values to Centroids array
    Centroids.append(cj_calc)
    #print(Centroids)
  return Centroids


# Update

# U^(k), U^(k+1)
# use weight_arr(uij^m) and C (Cj) as inputs
def updateWeights(weight_arr,Centroids):
  # set denominator equal to np.zero function which fills denominators with zeroes equal to n which is amount of entries in dataset
  denominator = np.zeros(n)
  # for loop for update 
  for i in range(k):
    distance = (data.iloc[:,:].values - Centroids[i])**2
    distance = np.sum(distance, axis=1)
    distance = np.sqrt(distance)
    denominator  = denominator + np.power(1/distance,1/(m-1))
  # for loop for update
  for i in range(k):
    distance = (data.iloc[:,:].values - Centroids[i])**2
    distance = np.sum(distance, axis=1)
    distance = np.sqrt(distance)
    weight_arr[:,i] = np.divide(np.power(1/distance,1/(m-1)),denominator)
  return weight_arr

#Calling Fuzzy C Means Calculation

def FuzzyCMeans():

  weight_arr = UIJMatrix()
  # for loop for all iterations
  for z in range(iterations):
    # find centroid
    C = CjCentroids(weight_arr)
    updateWeights(weight_arr,C)
    print(C)
  return (weight_arr,C)

final_weights,Centers = FuzzyCMeans()