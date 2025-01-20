# %%
import numpy as np
from scipy.interpolate import BSpline, splrep, UnivariateSpline
import adam
from scipy.signal import bspline
import pandas as pd

from sklearn.preprocessing import StandardScaler
import sparse
import time
from scipy.interpolate._bspl import evaluate_all_bspl
import multiprocessing as mp
import itertools
import random
import matplotlib.pyplot as plt

import sys
import argparse
import os

# %%
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
if __name__ == "__main__":
    set_random_seed(21)

# %% [markdown]
# ## IDLFM

# %%
def generate_data(n_patients, n_var, T, idx_x, idx_y, rank, k, N):
        D = N-k
        Fx = np.random.randn(n_var,rank)
        Fy = np.random.randn(rank)

        knots = np.array(list(range(1,N-2*k-1)))/(N-2*k-1)*T
        knots = np.insert(knots, 0, (k+1)*[-1])
        knots = np.insert(knots, N-k-1, (k+1)*[T+1])

        weights = np.random.randn(n_patients, rank, D)

        x_data = []
        y_data = []
        for i in range(n_patients):
            # Setting 1.1
            spl = [lambda t: 0.02*i*np.log(t+1), lambda t: 2*np.exp(-(t-60+10*i)/50*(t-60+10*i)+0.0000001) + 4*np.exp(-(t-70+10*i)/20*(t-70+10*i)+0.0000001) , lambda t: np.cos(0.12*np.pi*t) + 1]
           # Setting 1.2
           # spl = [lambda t: 0.02*np.log(t+1), lambda t: 2*np.exp(-(t-60+10*i)/50*(t-60+10*i)+0.0000001) + 4*np.exp(-(t-70+10*i)/20*(t-70+10*i)+0.0000001) , lambda t: np.cos(0.12*np.pi*t) + 1]
            for j in range(n_var):
                tmp = np.matmul(Fx[j,:], [spl[r](idx_x[i,j,:].data) for r in range(rank)])+ 0.5*np.random.randn(len(idx_x[i,j,:].data))
                x_data = np.concatenate((x_data, tmp))
            tmp = np.matmul(Fy, [spl[r](idx_y[i,0,:].data) for r in range(rank)])+ 0.5*np.random.randn(len(idx_y[i,0,:].data))
            y_data = np.concatenate((y_data, tmp))
        print(len(x_data))
        output_x = sparse.COO(idx_x.coords, x_data, shape = (n_patients, n_var, T))
        output_y = sparse.COO(idx_y.coords, y_data, shape = (n_patients, 1, T))


        return [output_x, output_y, knots, weights, Fx, Fy]

# %%
def xy_pred(weights, knots, F, n_patients, n_var, idx, rank, k):
        thetas = [[BSpline(knots, weights[i,r,:], k) for r in range(rank)] for i in range(n_patients)]
        data = []
        for i in range(n_patients):
            for j in range(n_var+1):
                tmp = np.matmul(F[j,:],[thetas[i][r](idx[i,j,:].data) for r in range(rank)])
                data = np.concatenate((data, tmp))
        output = sparse.COO(idx.coords, data,shape = (n_patients, n_var+1, idx.shape[2]))
        return output

# %%
def dlts(X, Y, n_patients, n_var, T, idx_x, idx_y, rank, k, N, lambda1 = 1, lambda2 = 1, Niter = 100, alpha = 0.001, ebs = 0.0001, l=1):
        beta1 = 0.9
        beta2 = 0.999
        m_w = 0
        m_F = 0
        v_w = 0
        v_F = 0

        D = N-k-1
        F = np.random.randn(n_var+1,rank)


        coords = np.copy(Y.coords)
        coords[1,:] = n_var
        coords = np.concatenate((X.coords,coords), axis = 1)
        data = np.concatenate((X.data,Y.data))
        xy = sparse.COO(coords, data, shape = (n_patients, n_var+1, T))


        knots = np.array(list(range(1,N-2*k-1)))/(N-2*k-1)*T
        knots = np.insert(knots, 0, (k+1)*[-1])
        knots = np.insert(knots, N-k-1, (k+1)*[T+1])

        coords = np.copy(idx_y.coords)
        coords[1,:] = n_var
        coords = np.concatenate((idx_x.coords,coords), axis = 1)
        data = np.concatenate((idx_x.data,idx_y.data))
        idx = sparse.COO(coords, data, shape = (n_patients, n_var+1, T))

        weights = np.random.randn(n_patients, rank, D)
        xy_hat = xy_pred(weights, knots, F, n_patients, n_var, idx, rank, k)


        nobs = len(data)
        S = np.sum((xy - xy_hat)**2)/nobs
        S_record = [S]
        print(S)

        K = np.zeros((D, T), dtype=np.float_)
        for t in range(T):
            xval = t
            if xval <= knots[k]:
                left = k
            else:
                left = np.searchsorted(knots, xval) - 1

            # fill a row
            bb = evaluate_all_bspl(knots*1.0, k, xval, left)
            K[left-k:left+1, t] = bb

        for itr in range(Niter):
            unique_t = np.sort(list(set(idx.data)))
            theta = np.tensordot(weights, K, axes = (2,0))
            grad_weights = np.tensordot(2*(xy_hat - xy), F, axes = (1, 0))
            grad_weights = np.tensordot(grad_weights, K, axes = (1, 1))
            ## Fused lasso for theta_i
            grad_pen = np.empty(weights.shape)
            trans_m = -np.eye(n_patients- 1)
            for i in range(n_patients):
                tmp = np.tensordot(np.insert(trans_m, i, np.ones(n_patients-1), axis=1), weights, axes = (1,0))
                tmp = np.sign(tmp)
                grad_pen[i] = np.sum(tmp, axis = 0)
            ##total variation for bspline
            jump = np.insert(-np.eye(D-1), 0, np.zeros(D-1), axis = 1)
            jump = np.insert(jump, D-1, np.zeros(D), axis = 0)
            jump = jump + np.eye(D)
            jump_m = jump
            if k > 1:
                for i in range(k):
                    jump_m = np.matmul(jump_m, jump)
            grad_pen1 = np.tensordot(weights, jump_m.T, axes = (2, 0))
            grad_pen1 = np.sign(grad_pen1)
            grad_pen1 = np.tensordot(grad_pen1[:,:,0:(D-k)], jump_m[0:(D-k)], axes = (2, 0))
            grad_weights = grad_weights + l*weights + lambda1 * grad_pen + lambda2 * grad_pen1
            grad_F = np.tensordot(2*(xy_hat - xy),theta, axes = ([0,2],[0,2]))
            grad_F += l*F

            m_w = beta1*m_w + (1-beta1) * grad_weights
            m_F = beta1*m_F + (1-beta1) * grad_F
            v_w = beta2 * v_w + (1-beta2) * grad_weights**2
            v_F = beta2 * v_F + (1-beta2) * grad_F**2
            mhat_w = m_w / (1-beta1)
            mhat_F = m_F / (1 - beta1)
            vhat_w = v_w / (1 - beta2)
            vhat_F = v_F / (1 - beta2)
            beta1 = beta1**(i+1)
            beta2 = beta2**(i+1)

            weights = weights - alpha * mhat_w /(np.sqrt(vhat_w) + 1e-8)
            F = F - alpha * mhat_F /(np.sqrt(vhat_F) + 1e-8)

            xy_hat = xy_pred(weights, knots, F, n_patients, n_var, idx, rank, k)
            S = np.sum((xy -xy_hat)**2)/nobs
            t = np.abs((S_record[-1] - S)/S_record[-1])
            if i > 10 and S >= np.max(S_record):
                print('Diverge')
                break
            if t < ebs:
                print(itr)
                print('Converge')
                S_record.append(S)
                break
            if itr%100 == 0:
                print(itr, S)

        print('Max iteration')
        X_hat = xy_hat[:, 0:n_var, :]
        Y_hat = xy_hat[:, n_var, :]
        Y_hat = Y_hat.reshape((n_patients, 1, T))

        return [weights, F, X_hat, Y_hat]

# %%
I = 33 # # of patients
J = 5 # # of variables for X
#T = 96 # max time
T = 1000
R = 3 # Rank
k = 3 # smooth degree for spline function
N = 300 # number of knots in spline function
D = N-k-1 # # of base spline function, determined by # of knots and smooth degree

# %%
#    ############# Multiresolution
idx_x = sparse.random((I, J-1, T), density = 0.8)
idx_y = sparse.random((I, 1, T), density = 0.2)
# Extract non-zero coordinates and values from idx_y
coords = idx_y.coords.T  # Shape (non-zero elements, 3)
data = idx_y.data  # Non-zero values
n_total = len(data)

# Shuffle indices to randomize allocation
indices = np.arange(n_total)
np.random.shuffle(indices)

# Split indices into 80% for training and 20% for calibration
n_train = int(n_total * 0.8)
train_indices = indices[:n_train]
cal_indices = indices[n_train:]

# Allocate non-zero values to idx_y_train and idx_y_cal
train_coords = coords[train_indices].T  # Transpose back to match sparse.COO
cal_coords = coords[cal_indices].T  # Transpose back to match sparse.COO
train_data = data[train_indices]
cal_data = data[cal_indices]

# Create sparse matrices for idx_y_train and idx_y_cal
idx_y_train = sparse.COO(train_coords, train_data, shape=idx_y.shape)
idx_y_cal = sparse.COO(cal_coords, cal_data, shape=idx_y.shape)

# Determine the coordinates of the zero values in idx_y
all_coords = np.array(np.meshgrid(*[np.arange(s) for s in idx_y.shape], indexing="ij")).T.reshape(-1, 3)
all_coords_set = set(map(tuple, all_coords))  # All possible coordinates
non_zero_coords_set = set(map(tuple, coords))  # Non-zero coordinates in idx_y
zero_coords_set = all_coords_set - non_zero_coords_set  # Find the zero positions

# Use all zero coordinates for idx_y_test
zero_coords = np.array(list(zero_coords_set)).T  # Transpose to match sparse.COO structure

# Create sparse matrix for idx_y_test
test_data = np.ones(zero_coords.shape[1])  # Set non-zero values to 1 (can be adjusted as needed)
idx_y_test = sparse.COO(zero_coords, test_data, shape=idx_y.shape)

# Print results for verification
print("idx_y_train:", idx_y_train)
print("idx_y_cal:", idx_y_cal)
print("idx_y_test:", idx_y_test)

# %%
data = generate_data(I, J-1, T, idx_x, idx_y_train, R, k, N)
print(data)

# %%
output_x = data[0]
output_y = data[1]
knots = data[2]
weights = data[3]
Fx = data[4]
Fy = data[5]

# %%
result = dlts(X = output_x, Y = output_y, n_patients = I, n_var = J, T = T, idx_x = idx_x, idx_y = idx_y_train, rank = R, k = k, N = N)
print(result)

# %% [markdown]
# ## CP_weighted_ACI
# ### 1. Setting
# I = 33 # # of patients
# 
# J = 5 # # of variables for X
# 
# T = 1000
# 
# R = 3 # Rank
# 
# k = 3 # smooth degree for spline function
# 
# N = 300 # number of knots in spline function
# 
# D = N-k-1 # # of base spline function, determined by # of knots and smooth degree
# 
# $\mathcal{D}_{train}=\left\{\left(i, j, t, Y_{i j}(t)\right) \in \mathcal{D}: j=1, \ldots, J\right\}$ with resolution=0.8 when $j=1, \dots, J-1$, and resolution=0.16 when $j=J$, $N_{train}=5280$
# 
# $\mathcal{D}_{cal}=\left\{\left(i, j, t, Y_{i j}(t)\right) \in \mathcal{D}: j=J\right\}$ with resolution=0.04, $N_{cal}=1320$

# %%
weights = result[0]
F = result[1]
X = data[0]
Fy = data[5]

# %% [markdown]
# ### Y_cal

# %%
y_cal_data = []
for i in range(I):
    spl = [lambda t: 0.02*i*np.log(t+1), lambda t: 2*np.exp(-(t-60+10*i)/50*(t-60+10*i)+0.0000001) + 4*np.exp(-(t-70+10*i)/20*(t-70+10*i)+0.0000001) , lambda t: np.cos(0.12*np.pi*t) + 1]
    tmp = np.matmul(Fy, [spl[r](idx_y_cal[i,0,:].data) for r in range(R)])+ 0.5*np.random.randn(len(idx_y_cal[i,0,:].data))
    y_cal_data = np.concatenate((y_cal_data, tmp))
Y_cal = sparse.COO(idx_y_cal.coords, y_cal_data, shape = (I, 1, T))
print(Y_cal)

# %% [markdown]
# ### Y_cal_hat

# %%
def y_pred(weights, knots, F, n_patients, idx_y, rank, k):
        thetas = [[BSpline(knots, weights[i,r,:], k) for r in range(rank)] for i in range(n_patients)]
        data = []
        for i in range(n_patients):
            #for j in range(n_var+1):
                tmp = np.matmul(F[0,:],[thetas[i][r](idx_y[i,0,:].data) for r in range(rank)])
                data = np.concatenate((data, tmp))
        output = sparse.COO(idx_y.coords, data,shape = (I, 1, idx_y.shape[2]))
        return output
Y_cal_hat = y_pred(weights, knots, F, I, idx_y_cal, R, k)
print(Y_cal_hat)

# %% [markdown]
# ### Y_test and Y_test_hat

# %%
y_test_data = []
for i in range(I):
    spl = [lambda t: 0.02*i*np.log(t+1), lambda t: 2*np.exp(-(t-60+10*i)/50*(t-60+10*i)+0.0000001) + 4*np.exp(-(t-70+10*i)/20*(t-70+10*i)+0.0000001) , lambda t: np.cos(0.12*np.pi*t) + 1]
    tmp = np.matmul(Fy, [spl[r](idx_y_test[i,0,:].data) for r in range(R)])+ 0.5*np.random.randn(len(idx_y_test[i,0,:].data))
    y_test_data = np.concatenate((y_test_data, tmp))
Y_test = sparse.COO(idx_y_test.coords, y_test_data, shape = (I, 1, T))
print(Y_test)

# %%
Y_test_hat = y_pred(weights, knots, F, I, idx_y_test, R, k)
print(Y_test_hat)

# %% [markdown]
# ### 2. nonconformity scores

# %%
def calculate_nonconformity_score(Y_cal, Y_cal_hat):
    assert Y_cal.shape == Y_cal_hat.shape, "Shapes of Y_cal and Y_cal_hat do not match."
    nonconformity_scores = np.abs(Y_cal.data - Y_cal_hat.data)
    
    return nonconformity_scores

nonconformity_scores = calculate_nonconformity_score(Y_cal, Y_cal_hat)
print("Nonconformity Scores:", nonconformity_scores)

# %%
def calculate_weights(Y_cal, t_test):
    Y_cal_times = Y_cal.coords[2, :].astype(np.int64)  # Time indices cast to signed integer
    # Calculate unnormalized weights
    #weights = np.exp(-np.abs(Y_cal_times - t_test))  # w_iJ(t; t_0)
    weights = np.abs(Y_cal_times - t_test) # w_iJ(t; t_0)
    # Normalize the weights (Equation 5)
    weights_normalized = weights / np.sum(weights)
    #print(f"t_test={t_test}, weights_normalized={weights_normalized[:10]}")  # First 10 weights
    return weights_normalized

def compute_weighted_quantile(nonconformity_scores, weights_normalized, alpha):
    # Sort nonconformity scores and corresponding weights in ascending order of scores
    sorted_indices = np.argsort(nonconformity_scores)
    sorted_scores = nonconformity_scores[sorted_indices]
    sorted_weights = weights_normalized[sorted_indices]
    # Compute the cumulative sum of the weights
    cumulative_weights = np.cumsum(sorted_weights)
    # Find the smallest score where cumulative weight >= 1 - alpha
    quantile_threshold = sorted_scores[np.searchsorted(cumulative_weights, 1 - alpha)]
    return quantile_threshold

# %%
def adjust_alpha_tilde_weighted_sACI(Y_cal, Y_cal_hat, nonconformity_scores, t_test, alpha, delta, alpha_tilde_int):
    # Extract calibration values and predictions
    Y_cal_values = Y_cal.data
    Y_cal_hat_values = Y_cal_hat.data

    # Calculate normalized weights for the given test time point
    weights_normalized = calculate_weights(Y_cal, t_test)

    # Initialize variables
    alpha_tilde = alpha_tilde_int
    while True:
        # Compute the weighted quantile threshold
        threshold = np.quantile(nonconformity_scores, 1 - alpha_tilde)
        # threshold = compute_weighted_quantile(nonconformity_scores, weights_normalized, alpha_tilde)
       # print(f"t_test={t_test}, alpha_tilde={alpha_tilde}, threshold={threshold}")

        # Calculate the weighted sum of points outside the interval
        weighted_sum_outside = 0
        for idx, (Y_k, Y_hat_k) in enumerate(zip(Y_cal_values, Y_cal_hat_values)):
            interval_lower = Y_hat_k - threshold
            interval_upper = Y_hat_k + threshold
            if Y_k < interval_lower or Y_k > interval_upper:
                weighted_sum_outside += weights_normalized[idx]

        # Compute the difference from alpha
        diff = abs(weighted_sum_outside - alpha)
        # Check convergence
        if diff <= delta:
           # print("Convergence achieved within the specified range.")
            break

        # Adjust alpha_tilde based on the weighted sum
        if weighted_sum_outside < alpha:
            alpha_tilde += delta
        elif weighted_sum_outside > alpha:
            alpha_tilde -= delta

        # Ensure alpha_tilde remains within [0, 1]
        alpha_tilde = max(0, min(1, alpha_tilde))

    return alpha_tilde

# %%
# Example usage
t_test = 900  # Test time point
alpha = 0.1
delta = 0.005
alpha_tilde_int = 0.2

# Adjust alpha_tilde for the given t_test
alpha_tilde = adjust_alpha_tilde_weighted_sACI(
    Y_cal, Y_cal_hat, nonconformity_scores, t_test, alpha, delta, alpha_tilde_int
)

# Print final results
print(f"Final adjusted alpha_tilde for t_test={t_test}: {alpha_tilde}")

# %%
def wACI_prediction_interval(Y_test_hat, nonconformity_scores, weights_normalized, alpha_tilde, t_test):

    # Compute the weighted quantile threshold
    quantile_threshold = compute_weighted_quantile(nonconformity_scores, weights_normalized, alpha_tilde)

    # Find the specific test point based only on T = t_test
    mask = Y_test_hat.coords[2, :] == t_test  # Select only time T = t_test

    if not np.any(mask):
        raise ValueError(f"Test point (T={t_test}) not found in Y_test_hat.")

    # Extract the predicted value for the test point
    predicted_value = Y_test_hat.data[np.where(mask)[0][0]]

    # Compute the prediction interval
    lower_bound = predicted_value - quantile_threshold
    upper_bound = predicted_value + quantile_threshold

    return lower_bound, upper_bound

# %%
def prediction_intervals_weighted_ACI_all(Y_cal, Y_cal_hat, Y_test_hat, nonconformity_scores, alpha, delta):
    prediction_intervals_dict = {}

    # Extract Y_test_hat coordinates and data
    Y_test_coords = Y_test_hat.coords
    Y_test_data = Y_test_hat.data

    # Extract Y_cal coordinates
    Y_cal_coords = Y_cal.coords[2, :].astype(np.int64)  # Time indices (T) in calibration set

    # Loop over test time points (T)
    unique_t_test = np.unique(Y_test_coords[2, :])  # Unique time indices in Y_test_hat
    for t_test in unique_t_test:
        # Compute weights for all calibration points for the current test time
        weights_normalized = calculate_weights(Y_cal, t_test)

        # Adjust alpha_tilde for the current test time point
        alpha_tilde = adjust_alpha_tilde_weighted_sACI(
            Y_cal, Y_cal_hat, nonconformity_scores, t_test, alpha, delta, alpha_tilde_int=0.2
        )

        # Compute the weighted threshold
        quantile_threshold = compute_weighted_quantile(nonconformity_scores, weights_normalized, alpha_tilde)

        # Filter test points for the current time (T)
        mask = Y_test_coords[2, :] == t_test
        indices = np.where(mask)[0]  # Indices of matching test points

        # Process all matching test points
        predicted_values = Y_test_data[indices]
        patient_indices = Y_test_coords[0, indices]

        # Compute prediction intervals for all matching points
        lower_bounds = predicted_values - quantile_threshold
        upper_bounds = predicted_values + quantile_threshold

        # Save intervals and alpha_tilde to the dictionary
        for i, lower, upper in zip(patient_indices, lower_bounds, upper_bounds):
            prediction_intervals_dict[(i, 1, t_test)] = {
                "interval": (lower, upper),
                "alpha_tilde": alpha_tilde,
            }

    return prediction_intervals_dict
# Usage
prediction_intervals_dict = prediction_intervals_weighted_ACI_all(
    Y_cal, Y_cal_hat, Y_test_hat, nonconformity_scores, alpha=0.1, delta=0.01 
)

# %%
def calculate_coverage_rate(Y_test, prediction_intervals_dict):
    total_points = 0
    covered_points = 0

    # Iterate through each value in Y_test
    for idx, value in enumerate(Y_test.data):
        # Extract the coordinates (I, J, T)
        I, _, T = Y_test.coords[:, idx]  # Skip J since it's always 1

        # Check if the prediction interval exists
        if (I, 1, T) in prediction_intervals_dict:  # Use 1 for J
            lower_bound, upper_bound = prediction_intervals_dict[(I, 1, T)]["interval"]

            # Check if the value is within the interval
            if lower_bound <= value <= upper_bound:
                covered_points += 1

        # Increment total points
        total_points += 1

    # Calculate the coverage rate
    coverage_rate = covered_points / total_points if total_points > 0 else 0

    return coverage_rate
coverage_rate = calculate_coverage_rate(Y_test, prediction_intervals_dict)
print(f"Coverage Rate: {coverage_rate:.4f}")


# %%
def calculate_average_interval_length(prediction_intervals_dict):
    total_length = 0
    total_intervals = len(prediction_intervals_dict)

    # Sum up the lengths of all intervals
    for key, value in prediction_intervals_dict.items():
        lower_bound, upper_bound = value["interval"]
        interval_length = upper_bound - lower_bound
        total_length += interval_length

    # Calculate the average length
    average_length = total_length / total_intervals if total_intervals > 0 else 0

    return average_length
average_length = calculate_average_interval_length(prediction_intervals_dict)
print(f"Average Length of Prediction Intervals: {average_length:.4f}")


# %%
# Initialize lists to store results
coverage_rates = []
average_lengths = []

# Number of iterations
num_iterations = 50
alpha_tilde_int = 0.15

for iteration in range(num_iterations):
    print(f"Iteration {iteration + 1}/{num_iterations}")

    # Reinitialize random seed for reproducibility
    np.random.seed(iteration)
    
    # Re-generate training, calibration, and test datasets
    idx_x = sparse.random((I, J-1, T), density=0.8)
    idx_y = sparse.random((I, 1, T), density=0.2)
    
    # Split data into training and calibration
    coords = idx_y.coords.T
    data = idx_y.data
    n_total = len(data)
    indices = np.arange(n_total)
    np.random.shuffle(indices)
    n_train = int(n_total * 0.8)
    train_indices = indices[:n_train]
    cal_indices = indices[n_train:]

    train_coords = coords[train_indices].T
    cal_coords = coords[cal_indices].T
    train_data = data[train_indices]
    cal_data = data[cal_indices]
    idx_y_train = sparse.COO(train_coords, train_data, shape=idx_y.shape)
    idx_y_cal = sparse.COO(cal_coords, cal_data, shape=idx_y.shape)

    # Generate data
    data = generate_data(I, J-1, T, idx_x, idx_y_train, R, k, N)
    output_x, output_y, knots, weights, Fx, Fy = data

    # Perform model fitting
    result = dlts(output_x, output_y, I, J, T, idx_x, idx_y_train, R, k, N)
    weights, F = result[0], result[1]

    # Generate Y_cal, Y_cal_hat, Y_test, and Y_test_hat
    y_cal_data = []
    for i in range(I):
        spl = [lambda t: 0.02 * i * np.log(t + 1), 
               lambda t: 2 * np.exp(-(t - 60 + 10 * i) / 50 * (t - 60 + 10 * i)) + 4 * np.exp(-(t - 70 + 10 * i) / 20 * (t - 70 + 10 * i)),
               lambda t: np.cos(0.12 * np.pi * t) + 1]
        tmp = np.matmul(Fy, [spl[r](idx_y_cal[i, 0, :].data) for r in range(R)]) + 0.5 * np.random.randn(len(idx_y_cal[i, 0, :].data))
        y_cal_data = np.concatenate((y_cal_data, tmp))
    Y_cal = sparse.COO(idx_y_cal.coords, y_cal_data, shape=(I, 1, T))
    Y_cal_hat = y_pred(weights, knots, F, I, idx_y_cal, R, k)

    y_test_data = []
    for i in range(I):
        spl = [lambda t: 0.02 * i * np.log(t + 1), 
               lambda t: 2 * np.exp(-(t - 60 + 10 * i) / 50 * (t - 60 + 10 * i)) + 4 * np.exp(-(t - 70 + 10 * i) / 20 * (t - 70 + 10 * i)),
               lambda t: np.cos(0.12 * np.pi * t) + 1]
        tmp = np.matmul(Fy, [spl[r](idx_y_test[i, 0, :].data) for r in range(R)]) + 0.5 * np.random.randn(len(idx_y_test[i, 0, :].data))
        y_test_data = np.concatenate((y_test_data, tmp))
    Y_test = sparse.COO(idx_y_test.coords, y_test_data, shape=(I, 1, T))
    Y_test_hat = y_pred(weights, knots, F, I, idx_y_test, R, k)
    
    nonconformity_scores = calculate_nonconformity_score(Y_cal, Y_cal_hat)

    # Compute prediction intervals
    prediction_intervals_dict = prediction_intervals_weighted_ACI_all(
        Y_cal, Y_cal_hat, Y_test_hat, nonconformity_scores, alpha=0.1, delta=0.01
    )

    # Evaluate coverage rate and average length
    coverage_rate = calculate_coverage_rate(Y_test, prediction_intervals_dict)
    average_length = calculate_average_interval_length(prediction_intervals_dict)

    # Store results
    coverage_rates.append(coverage_rate)
    average_lengths.append(average_length)

# Compute and display summary statistics
print(f"Mean Coverage Rate: {np.mean(coverage_rates):.4f} ± {np.std(coverage_rates):.4f}")
print(f"Mean Average Interval Length: {np.mean(average_lengths):.4f} ± {np.std(average_lengths):.4f}")


# %%
# Initialize lists to store results
coverage_rates = []
average_lengths = []

# Number of iterations
num_iterations = 50
alpha_tilde_int = 0.15

for iteration in range(num_iterations):
    print(f"Iteration {iteration + 1}/{num_iterations}")

    # Reinitialize random seed for reproducibility
    np.random.seed(iteration)
    
    # Fixed multiresolution
    a = np.ones((T))
    b = np.zeros((T))
    b[range(0, T, 2)] = 1
    idx_x = np.stack((J-1-int(J/4))*[a]+int(J/4)*[b], axis = 0)
    idx_x = np.repeat(idx_x.reshape(1,J-1,T), I, axis=0)
    idx_x = sparse.COO.from_numpy(idx_x)
    b = np.zeros((T))
    b[range(0, T, 4)] = 1
    idx_y = np.repeat(b.reshape(1,1,T), I, axis=0)
    idx_y = sparse.COO.from_numpy(idx_y)

    # Extract non-zero coordinates and values from idx_y
    coords = idx_y.coords.T  # Shape (non-zero elements, 3)
    data = idx_y.data  # Non-zero values
    n_total = len(data)

    # Shuffle indices to randomize allocation
    indices = np.arange(n_total)
    np.random.shuffle(indices)

    # Split indices into 80% for training and 20% for calibration
    n_train = int(n_total * 0.8)
    train_indices = indices[:n_train]
    cal_indices = indices[n_train:]

    # Allocate non-zero values to idx_y_train and idx_y_cal
    train_coords = coords[train_indices].T  # Transpose back to match sparse.COO
    cal_coords = coords[cal_indices].T  # Transpose back to match sparse.COO
    train_data = data[train_indices]
    cal_data = data[cal_indices]

    # Create sparse matrices for idx_y_train and idx_y_cal
    idx_y_train = sparse.COO(train_coords, train_data, shape=idx_y.shape)
    idx_y_cal = sparse.COO(cal_coords, cal_data, shape=idx_y.shape)

    # Determine the coordinates of the zero values in idx_y
    all_coords = np.array(np.meshgrid(*[np.arange(s) for s in idx_y.shape], indexing="ij")).T.reshape(-1, 3)
    all_coords_set = set(map(tuple, all_coords))  # All possible coordinates
    non_zero_coords_set = set(map(tuple, coords))  # Non-zero coordinates in idx_y
    zero_coords_set = all_coords_set - non_zero_coords_set  # Find the zero positions

    # Use all zero coordinates for idx_y_test
    zero_coords = np.array(list(zero_coords_set)).T  # Transpose to match sparse.COO structure

    # Create sparse matrix for idx_y_test
    test_data = np.ones(zero_coords.shape[1])  # Set non-zero values to 1 (can be adjusted as needed)
    idx_y_test = sparse.COO(zero_coords, test_data, shape=idx_y.shape)

    # Generate data
    data = generate_data(I, J-1, T, idx_x, idx_y_train, R, k, N)
    output_x, output_y, knots, weights, Fx, Fy = data

    # Perform model fitting
    result = dlts(output_x, output_y, I, J, T, idx_x, idx_y_train, R, k, N)
    weights, F = result[0], result[1]

    # Generate Y_cal, Y_cal_hat, Y_test, and Y_test_hat
    y_cal_data = []
    for i in range(I):
        spl = [lambda t: 0.02 * i * np.log(t + 1), 
               lambda t: 2 * np.exp(-(t - 60 + 10 * i) / 50 * (t - 60 + 10 * i)) + 4 * np.exp(-(t - 70 + 10 * i) / 20 * (t - 70 + 10 * i)),
               lambda t: np.cos(0.12 * np.pi * t) + 1]
        tmp = np.matmul(Fy, [spl[r](idx_y_cal[i, 0, :].data) for r in range(R)]) + 0.5 * np.random.randn(len(idx_y_cal[i, 0, :].data))
        y_cal_data = np.concatenate((y_cal_data, tmp))
    Y_cal = sparse.COO(idx_y_cal.coords, y_cal_data, shape=(I, 1, T))
    Y_cal_hat = y_pred(weights, knots, F, I, idx_y_cal, R, k)

    y_test_data = []
    for i in range(I):
        spl = [lambda t: 0.02 * i * np.log(t + 1), 
               lambda t: 2 * np.exp(-(t - 60 + 10 * i) / 50 * (t - 60 + 10 * i)) + 4 * np.exp(-(t - 70 + 10 * i) / 20 * (t - 70 + 10 * i)),
               lambda t: np.cos(0.12 * np.pi * t) + 1]
        tmp = np.matmul(Fy, [spl[r](idx_y_test[i, 0, :].data) for r in range(R)]) + 0.5 * np.random.randn(len(idx_y_test[i, 0, :].data))
        y_test_data = np.concatenate((y_test_data, tmp))
    Y_test = sparse.COO(idx_y_test.coords, y_test_data, shape=(I, 1, T))
    Y_test_hat = y_pred(weights, knots, F, I, idx_y_test, R, k)
    
    nonconformity_scores = calculate_nonconformity_score(Y_cal, Y_cal_hat)

    # Compute prediction intervals
    prediction_intervals_dict = prediction_intervals_weighted_ACI_all(
        Y_cal, Y_cal_hat, Y_test_hat, nonconformity_scores, alpha=0.1, delta=0.01
    )

    # Evaluate coverage rate and average length
    coverage_rate = calculate_coverage_rate(Y_test, prediction_intervals_dict)
    average_length = calculate_average_interval_length(prediction_intervals_dict)

    # Store results
    coverage_rates.append(coverage_rate)
    average_lengths.append(average_length)

# Compute and display summary statistics
print(f"Mean Coverage Rate: {np.mean(coverage_rates):.4f} ± {np.std(coverage_rates):.4f}")
print(f"Mean Average Interval Length: {np.mean(average_lengths):.4f} ± {np.std(average_lengths):.4f}")


