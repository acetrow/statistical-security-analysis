# Cybersecurity Risk Analysis & Optimization

A Python project implementing statistical modeling, probability analysis, and linear programming for cybersecurity risk assessment and resource optimization. This project demonstrates advanced applications of probability distributions, Bayesian inference, and optimization techniques.

Completed as part of an assessed exercise covering computational statistics and operations research methods.

---

## Features

### Task 1: Risk Assessment with Statistical Distributions
- Calculates probabilities using **Triangular Distribution** (CDF, mean, median)
- Computes discrete distribution parameters (mean and variance)
- Simulates cyber attack impacts using **Log-Normal** and **Pareto distributions**
- Calculates **Annual Loss Expectancy (ALE)** using:
  - ALE = ARO × SLE
  - SLE = AV × EF
  - Where ARO (Annual Rate of Occurrence), AV (Asset Value), EF (Exposure Factor)

### Task 2: Joint Probability & Bayesian Inference
- Analyzes joint probability distributions for discrete random variables
- Computes marginal and conditional probabilities
- Applies **Bayes' Theorem** to calculate posterior probabilities
- Handles complex probability constraints (e.g., P(X + Y ≤ 10))

### Task 3: Security Resource Optimization
- Performs **multivariate linear regression** using `scipy.optimize.curve_fit`
- Models security effectiveness and maintenance load as functions of resource allocation
- Solves **constrained linear programming** problems using `scipy.optimize.linprog`
- Optimizes resource distribution subject to:
  - Minimum security effectiveness thresholds
  - Maximum maintenance load constraints
  - Resource capacity bounds

---

## Project Structure

- `Task1` — Statistical risk modeling with triangular, discrete, log-normal, and Pareto distributions
- `Task2` — Joint probability analysis and Bayesian inference
- `Task3` — Linear regression modeling and constrained optimization
- `dhCheck_Task*.py` — Validation modules (not included in repository)

---

## Requirements

Python 3.7+, numpy, scipy

Install dependencies:

pip install numpy scipy

---

## Usage

Each task is implemented as a standalone function that can be imported and called:

import numpy as np
from your_module import Task1, Task2, Task3

# Task 1: Risk Assessment
result1 = Task1(a, b, c, point1, number_set, prob_set, num, point2, mu, sigma, xm, alpha, point3, point4)
prob1, mean_t, median_t, mean_d, var_d, prob2, prob3, ale = result1

# Task 2: Probability Analysis
result2 = Task2(num, table, probs)
prob1, prob2, prob3 = result2

# Task 3: Optimization
result3 = Task3(x, y, z, x_initial, c, x_bound, se_bound, ml_bound)
weights_b, weights_d, x_add = result3

---

## Key Algorithms

### Triangular Distribution CDF

P(X ≤ x) = (x - a)² / ((b - a)(c - a)) for a < x ≤ c
P(X ≤ x) = 1 - (b - x)² / ((b - a)(b - c)) for c < x < b

### Annual Loss Expectancy (ALE)

ALE = ARO × SLE
SLE = AV × EF

### Bayes' Theorem Application

P(Y=8|T) = P(T|Y=8) × P(Y=8) / P(T)

### Linear Programming Formulation

minimize: c^T × x_add
subject to: 
- Safety effectiveness ≥ threshold
- Maintenance load ≤ limit
- 0 ≤ x_add ≤ capacity bounds

---

## Example Output

**Task 1**: Returns tuple of (prob1, mean, median, discrete_mean, variance, prob2, prob3, ALE)

**Task 2**: Returns tuple of (P(3 ≤ X ≤ 4), P(X + Y ≤ 10), P(Y=8|T))

**Task 3**: Returns tuple of (regression_weights_b, regression_weights_d, optimal_resource_allocation)

---

## License

This project is released under the MIT License.