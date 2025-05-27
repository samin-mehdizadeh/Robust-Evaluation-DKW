import pandas as pd
import numpy as np
from scipy.optimize import root_scalar
import cvxpy as cp
from scipy.special import rel_entr

def kl(t):
  return t * np.log(t)

def kl_inverse(y):
    def equation(t):
        return t * np.log(t) - y
    result = root_scalar(equation, bracket=(1e-10, y), method='brentq')

    if result.converged:
        return result.root
    else:
        raise RuntimeError("Failed to find the inverse.")

def chi_square(t):
    return (t - 1) ** 2

def chi_square_inverse(y):
    return 1 + np.sqrt(y)

def calcualte_fdivergence_optimization(losses,lambda_,epsilon,f_divergence):
    K = len(losses)
    n_k = 1000
    delta = 0.05
    if(f_divergence == 'kl'):
      Lambda = kl_inverse(epsilon/delta)
      c1 = Lambda - 1
      c2 = (kl(Lambda)-kl(1/Lambda))/2
    elif(f_divergence == 'chi-square'):
      Lambda = chi_square_inverse(epsilon/delta)
      c1 = Lambda - 1
      c2 = (chi_square(Lambda))/2
 
    constant = np.sqrt(np.log((K + 2) / delta) / (2 * n_k))
    QV_hat = losses
    thresholds = lambda_ - constant
    alpha = cp.Variable(K)
    indicator = (QV_hat >= thresholds).astype(float)
    objective = cp.Maximize(cp.sum(cp.multiply(alpha, indicator)) / K)

    f_divergence_constraint = {
        "chi-square": cp.sum(cp.square(1 - alpha)) / K <= epsilon + c2 * np.sqrt(np.log((K) / delta) / K),
        "kl": cp.sum(cp.kl_div(alpha, 1) + alpha - 1) / K <=  epsilon + c2 * np.sqrt(np.log((K) / delta) / K)
    }

    constraints = [
        0 <= alpha,
        alpha <= Lambda,
        cp.abs(cp.sum(alpha) / K - 1) <= c1 * np.sqrt(np.log((K) / delta) / K),
        f_divergence_constraint[f_divergence]
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()
    return problem.value

def calculate_meta_chisquare_divergence(samples1, samples2, num_bins=10):
    hist1, bin_edges = np.histogram(samples1, bins=num_bins, range=(min(samples1.min(), samples2.min()), max(samples1.max(), samples2.max())), density=True)
    hist2, _ = np.histogram(samples2, bins=bin_edges, density=True)
    dx = np.diff(bin_edges)
    epsilon = min([x for x in hist2 if x!=0])
    hist2 = np.maximum(hist2, epsilon)
    chi_square = np.sum((hist1 - hist2) ** 2 / hist2 * dx)
    return chi_square

def calculate_meta_kl_divergence(samples1, samples2, bins=10):
    hist1, bin_edges = np.histogram(samples1, bins=bins,range=(min(samples1.min(), samples2.min()), max(samples1.max(), samples2.max())), density=True)
    hist2, _ = np.histogram(samples2, bins=bin_edges,density=True)
    dx = np.diff(bin_edges)
    epsilon = min([x for x in hist2 if x!=0])
    hist2 = np.maximum(hist2, epsilon)
    kl_div = np.sum(rel_entr(hist1, hist2) * dx)
    return kl_div