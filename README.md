# Cybersecurity Risk Analysis & Resource Optimization

## Overview

This project demonstrates how statistical modeling, Bayesian inference, and optimization techniques can be applied to assess cybersecurity risk and optimize security resource allocation under uncertainty.

Using Python and scientific computing libraries, the project simulates cyber attack scenarios, estimates expected financial losses, and recommends optimal security investment strategies subject to real-world constraints.

## Problem Statement

Organizations face increasing cyber threats while operating under limited security budgets. Decision-makers must balance:

- Uncertain attack likelihoods
- Potential financial impact
- Resource and capacity constraints

This project addresses these challenges by combining probabilistic modeling with constrained optimization to support data-driven security decisions.

## Approach

### 1. Statistical Risk Modeling

- Modeled cyber incident likelihoods using:
  - Triangular distribution
  - Log-Normal distribution
  - Pareto distribution
- Computed key risk metrics including:
  - Mean and variance
  - Probability of loss events
  - Annual Loss Expectancy (ALE)

### 2. Bayesian Inference

- Analyzed joint, marginal, and conditional probabilities
- Applied Bayes' Theorem to update risk estimates based on new evidence
- Evaluated complex probability constraints (e.g., combined event thresholds)

### 3. Security Resource Optimization

- Built regression models to capture security effectiveness and maintenance load
- Formulated a constrained linear programming problem to:
  - Maximize security effectiveness
  - Minimize maintenance and operational cost
  - Respect resource capacity limits
- Solved optimization using `scipy.optimize`

## Key Outcomes

- Quantified expected financial losses under different cyber threat scenarios
- Demonstrated how Bayesian updates improve risk estimation accuracy
- Identified optimal security resource allocations under multiple constraints
- Showed how probabilistic models can support strategic security planning

## Technologies Used

- Python 3
- NumPy
- SciPy
- Linear Programming
- Statistical Distributions
- Bayesian Probability

## Project Structure

- `Task1` – Statistical risk modeling and loss estimation
- `Task2` – Joint probability analysis and Bayesian inference
- `Task3` – Regression modeling and constrained optimization

## How to Run

Install dependencies:
```bash
pip install numpy scipy
```

Example usage:
```python
from tasks import Task1, Task2, Task3

# Risk assessment
result1 = Task1(...)

# Probability analysis
result2 = Task2(...)

# Optimization
result3 = Task3(...)
```

## What I Learned

- Translating probabilistic theory into practical risk models
- Applying Bayesian reasoning to real-world uncertainty
- Designing and solving constrained optimization problems
- Writing modular, reusable Python code for analytical workflows

## Notes

This project was originally developed as part of a computational statistics module and extended to emphasize real-world cybersecurity and decision-making applications.

## License

MIT License
