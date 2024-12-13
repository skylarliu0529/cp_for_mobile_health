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

import sys
import argparse
import os

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
if __name__ == "__main__":
    set_random_seed(20)
################################## IDLFM ###########################
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

def xy_pred(weights, knots, F, n_patients, n_var, idx, rank, k):
        thetas = [[BSpline(knots, weights[i,r,:], k) for r in range(rank)] for i in range(n_patients)]
        data = []
        for i in range(n_patients):
            for j in range(n_var+1):
                tmp = np.matmul(F[j,:],[thetas[i][r](idx[i,j,:].data) for r in range(rank)])
                data = np.concatenate((data, tmp))
        output = sparse.COO(idx.coords, data,shape = (n_patients, n_var+1, idx.shape[2]))
        return output

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
######################Settings########################
I = 33 # # of patients
J = 5 # # of variables for X
#T = 96 # max time
T = 1000
R = 3 # Rank
k = 3 # smooth degree for spline function
N = 300 # number of knots in spline function
D = N-k-1 # # of base spline function, determined by # of knots and smooth degree

############### Multiresolution#########################
idx_x = sparse.random((I, J-1, T), density = 0.8)
# idx_y = sparse.random((I, 1, T), density = 0.2)
idx_y_train = sparse.random((I, 1, T), density = 0.2)
# idx_y_train = randomly choose 0.8 of idx_y
# idx_y_cal = the remaining 0.2 of idx_y
idx_y_cal = sparse.random((I, 1, T), density = 0.2)

data = generate_data(I, J-1, T, idx_x, idx_y_train, R, k, N)
print(data)
output_x = data[0]
output_y = data[1]
knots = data[2]
weights = data[3]
Fx = data[4]
Fy = data[5]
result = dlts(X = output_x, Y = output_y, n_patients = I, n_var = J, T = T, idx_x = idx_x, idx_y = idx_y_train, rank = R, k = k, N = N)
print(result)
#########################Conformal Prediction#################
weights = result[0]
F = result[1]
X = data[0]
Fy = data[5]
############Y_cal and Y_cal_hat############
y_cal_data = []
for i in range(I):
    spl = [lambda t: 0.02*i*np.log(t+1), lambda t: 2*np.exp(-(t-60+10*i)/50*(t-60+10*i)+0.0000001) + 4*np.exp(-(t-70+10*i)/20*(t-70+10*i)+0.0000001) , lambda t: np.cos(0.12*np.pi*t) + 1]
    tmp = np.matmul(Fy, [spl[r](idx_y_test[i,0,:].data) for r in range(R)])+ 0.5*np.random.randn(len(idx_y_test[i,0,:].data))
    y_cal_data = np.concatenate((y_cal_data, tmp))
Y_cal = sparse.COO(idx_y_test.coords, y_cal_data, shape = (I, 1, T))
print(Y_cal)

def y_pred(weights, knots, F, n_patients, idx_y, rank, k):
        thetas = [[BSpline(knots, weights[i,r,:], k) for r in range(rank)] for i in range(n_patients)]
        data = []
        for i in range(n_patients):
            #for j in range(n_var+1):
                tmp = np.matmul(F[0,:],[thetas[i][r](idx_y[i,0,:].data) for r in range(rank)])
                data = np.concatenate((data, tmp))
        output = sparse.COO(idx_y.coords, data,shape = (I, 1, idx_y.shape[2]))
        return output
Y_cal_hat = y_pred(weights, knots, F, I, idx_y_test, R, k)
print(Y_cal_hat)
####################### nonconformity scores #####################
def calculate_nonconformity_score(Y_cal, Y_cal_hat):
    assert Y_cal.shape == Y_cal_hat.shape, "Shapes of Y_cal and Y_cal_hat do not match."
    nonconformity_scores = np.abs(Y_cal.data - Y_cal_hat.data)
    
    return nonconformity_scores

nonconformity_scores = calculate_nonconformity_score(Y_cal, Y_cal_hat)
print("Nonconformity Scores:", nonconformity_scores)

###############Simplified ACI###################
def calculate_alpha_tilde_simplified_ACI(Y_cal, Y_cal_hat, nonconformity_scores, alpha, delta, alpha_tilde_int):
    Y_cal_values = Y_cal.data  
    Y_cal_hat_values = Y_cal_hat.data 

    alpha_tilde = alpha_tilde_int  
    N_cal = len(Y_cal_values)  
    iteration = 0  

    alpha_tilde_history = []
    threshold_history = []
    outside_rate_history = []

    while True:
        iteration += 1 
        alpha_tilde_history.append(alpha_tilde) 
        threshold = np.quantile(nonconformity_scores, 1 - alpha_tilde)
        threshold_history.append(threshold)  
        outside_count = 0
        for Y_k, Y_hat_k in zip(Y_cal_values, Y_cal_hat_values):
            interval_lower = Y_hat_k - threshold
            interval_upper = Y_hat_k + threshold
            if Y_k < interval_lower or Y_k > interval_upper:
                outside_count += 1
        
        outside_rate = outside_count / N_cal
        outside_rate_history.append(outside_rate)
        print(f"Iteration {iteration}:")
        print(f"  alpha_tilde = {alpha_tilde}")
        print(f"  Threshold = {threshold}")
        print(f"  Outside count = {outside_count}")
        print(f"  Outside rate = {outside_rate}")
        
        if alpha - delta <= outside_rate <= alpha + delta:
            print("Convergence achieved within the specified range.")
            break
        if outside_rate < alpha - delta:
            alpha_tilde += delta
        elif outside_rate > alpha + delta:
            alpha_tilde -= delta
        alpha_tilde = max(0, min(1, alpha_tilde))
    
    return alpha_tilde, alpha_tilde_history, threshold_history, outside_rate_history

# Example 
alpha = 0.1  
delta = 0.005  
nonconformity_scores = calculate_nonconformity_score(Y_cal, Y_cal_hat) 
alpha_tilde, alpha_tilde_history, threshold_history, outside_rate_history = calculate_alpha_tilde_simplified_ACI(Y_cal, Y_cal_hat, nonconformity_scores, alpha, delta, 0.2)

print("Final adjusted significance level (alpha_tilde):", alpha_tilde)
##################Predict Interval###################
def prediction_interval(Y_cal_hat, nonconformity_scores, alpha_tilde):
    threshold = np.quantile(nonconformity_scores, 1 - alpha_tilde)
    prediction_intervals_dict = {}
    for idx, value in enumerate(Y_cal_hat.data):
        I, J, T = Y_cal_hat.coords[:, idx]
        lower_bound = value - threshold
        upper_bound = value + threshold
        prediction_intervals_dict[(I, J, T)] = (lower_bound, upper_bound)
    for (I, J, T), interval in prediction_intervals_dict.items():
        print(f"Prediction Interval for (I={I+1}, J={J+1}, T={T+1}): {interval}")
    
    return prediction_intervals_dict
prediction_intervals_dict = prediction_interval(Y_cal_hat, nonconformity_scores, alpha_tilde)
