# -- START OF YOUR CODERUNNER SUBMISSION CODE
# INCLUDE ALL YOUR IMPORTS HERE
import math
import numpy as np


from dhCheck_Task1 import dhCheckCorrectness
def Task1(a, b, c, point1, number_set, prob_set, num, point2, mu, sigma, xm, alpha, point3, point4):
    #formula from triangular distribution
    if (point1 <= a):
        prob1 = 0
        
    elif (a < point1 <= c ):
        prob1 = ((point1 - a)**2)/((b-a)*(c-a))
    
    elif (c < point1 < b):
        prob1 = 1- (((b-point1)**2)/((b-a)*(b-c)))
    
    elif b<= point1:
        point1 = 1
    
    
    MEAN_t = (a+b+c)/3
    
    if (c >= (a+b)/2):
        MEDIAN_t = a + math.sqrt((b-a)*(c-a)/2)
    elif (c <= (a+b)/2):
        MEDIAN_t = b - math.sqrt((b-a)*(b-c)/2)
    
    #E(X)=μ=∑xP(x)
    MEAN_d = 0
    for i in range(len(number_set)):
        MEAN_d += number_set[i] * prob_set[i]
    #Var(X) = E(X^2) - μ^2
    MEAN_dPower2 = 0 # initialize E(X^2)
    for i in range(len(number_set)):
        MEAN_dPower2 += (number_set[i]**2) * (prob_set[i])
    
    VARIANCE_d = MEAN_dPower2 - (MEAN_d)**2
    
    totalImpacts = []
    #find the total impacts
    for _ in range(num):
        
        impact_A = np.random.lognormal(mu, sigma)
        
        impact_B = np.random.pareto(alpha) * xm
        
        impactAandB = impact_A + impact_B
        totalImpacts.append(impactAandB)
    
    totalImpacts = np.array(totalImpacts)
    
    # calculate prob2
    prob2 = np.sum(totalImpacts > point2) / num
    #calculate prob3
    prob3 = np.sum((totalImpacts > point3) & (totalImpacts < point4)) / num

    #ALE = ARO * SLE (ARO = MEAN_d)
    #SLE = AV * EF (AV = MEDIAN_t and EF = prob2)
    
    #initialize the variables
    AV = MEDIAN_t 
    EF = prob2
    ARO = MEAN_d
    
    SLE = AV * EF
    ALE = ARO * SLE
    
    return (prob1, MEAN_t, MEDIAN_t, MEAN_d, VARIANCE_d, prob2, prob3, ALE)

# -- END OF YOUR CODERUNNER SUBMISSION CODE
# -- START OF YOUR CODERUNNER SUBMISSION CODE
# INCLUDE ALL YOUR IMPORTS HERE

from dhCheck_Task2 import dhCheckCorrectness
def Task2(num, table, probs):
    # TODO
    # row with values a, b, c, d = table [0] = (Y = 6)
    Y6 = table[0]  # Y = 6 corresponds to row 0
    
    # row with values e, f, g, h = table [1] = (Y = 7)    
    Y7 = table[1]
    
    # row i, j, k, l = table [2] = (Y = 8)
    Y8 = table[2]  # Y = 8 correspondsa to row 2
    
    #prob1 = P(3 ≤ X ≤ 4) = bc, fg, jk
    prob1 = (Y6[1] + Y6[2] +   # b and c where y = 6
            Y7[1] + Y7[2] +   # f and g where y = 7
            Y8[1] + Y8[2]) / num  # j and k where y = 8
    
    # prob2 = P(X + Y ≤ 10)
    #P(X + Y <= 10) = x+y = 2+6, 3+6, 4+6, [Y =6]
                            #2+7, 3+7 [Y = 7]
                            #2+8 [Y = 8]
    prob2 = (Y6[0] + Y6[1] + Y6[2] +
            Y7[0] + Y7[1] +
            Y8[0])/num
        
    PX = [probs[0], probs[1], probs[2], probs[3]]  # P(T|X=2) to P(T|X=5)
    PY = [probs[4], probs[5]]  # P(T|Y=6), P(T|Y=7)

    #prob3
    
    #given variables
    PX2 = probs[0] 
    PX3 = probs[1] 
    PX4 = probs[2] 
    PX5 = probs[3] 
    PY6 = probs[4] 
    PY7 = probs[5] 
    
    X2 = table[0][0] + table[1][0] + table[2][0]  # a + e + i
    X3 = table[0][1] + table[1][1] + table[2][1]  # b + f + j
    X4 = table[0][2] + table[1][2] + table[2][2]  # c + g + k
    X5 = table[0][3] + table[1][3] + table[2][3]  # d + h + l
    # Y7 AND Y8 were initialized in the earlier of the code
    
    #after deriving, the formula of P(T) :-
    
    PT = PX2 * X2 + PX3 * X3 + PX4 * X4 + PX5 * X5
    #PT = PY8 * Y8 + PY7 * Y7 + PY6 * Y6
    
    #combining both denominators, we can find P(T|Y=8)
    #To find P(T|Y=8)
    #P(T|Y=8) = (PX2 * X2 + PX3 *X3 + PX4 * X4 + PX5 * X5 - (PY7 *Y7) - (PY6 * Y6))/Y8
    
    #PY8 = (PX2 * X2 + PX3 *X3 + PX4 * X4 + PX5 * X5 - (PY7 *Y7) - (PY6 * Y6)) / Y8
    PY8 = (PT - (PY7 * sum(Y7)) - (PY6 * sum(Y6))) / sum(Y8)

    # since we have all the variables, now we can find prob3
    
    #prob3 = (PY8 * Y8) / (PY8 * Y8 + PY7 * Y7)
    prob3 = (PY8 * sum(Y8)) / (PT)
    
    return (prob1, prob2, prob3)

# -- END OF YOUR CODERUNNER SUBMISSION CODE
# -- START OF YOUR CODERUNNER SUBMISSION CODE
# INCLUDE ALL YOUR IMPORTS HERE
import numpy as np
from scipy.optimize import linprog, curve_fit
from dhCheck_Task3 import dhCheckCorrectness


from dhCheck_Task3 import dhCheckCorrectness
def Task3(x, y, z, x_initial, c, x_bound, se_bound, ml_bound):
    # TODO
    
    #1) Linear reg for safeguard y================================================
    
    def linear_model(X, b0, b1, b2, b3, b4):
        # X is (4, N) so X[0] are x1, X[1] are x2 etc
        return b0 + b1*X[0] + b2*X[1] + b3*X[2] + b4*X[3]

    # for y
    optParam_b, _ = curve_fit(linear_model, x, y)
    weights_b = np.array(optParam_b)  # which is [b0, b1, b2, b3, b4]
    
    #================================================================
    
    #2) Linear reg for maintenance load z ==================================================
    
    optParam_d, _ = curve_fit(linear_model, x, z)
    weights_d = np.array(optParam_d)  # which is [d0, d1, d2, d3, d4]

    # unpack to make it easier
    b0, b1, b2, b3, b4 = weights_b
    d0, d1, d2, d3, d4 = weights_d
    
    #================================================================
    
    #3) Linear prog (find x_add) ==================================================
    
    # 3.1) sum(b_i * x_add_i) >= ..
    lhs_se = np.array([-b1, -b2, -b3, -b4])
    rhs_se = - (se_bound - b0 - (b1*x_initial[0] + b2*x_initial[1] + b3*x_initial[2] + b4*x_initial[3]))
    
    # 3.2) sum(d_i * x_add_i) <= ..
    lhs_ml = np.array([d1, d2, d3, d4])
    rhs_ml = ml_bound - d0 - (d1*x_initial[0] + d2*x_initial[1] + d3*x_initial[2] + d4*x_initial[3])
    
    
    # Combine those into A_ub and b_ub for linprog
    A_ub = np.vstack([lhs_se, lhs_ml])   #(2,4)
    b_ub = np.array([rhs_se, rhs_ml])    #(2,)
    
    # 3.3) 0 <= x_add[i] <= x_bound[i] - x_initial[i]
    bounds = []
    for i in range(4):
        lower_bound = 0
        upper_bound = x_bound[i] - x_initial[i]
        bounds.append((lower_bound,upper_bound))
        
    # 3.4) solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    x_add = res.x
    
    #================================================================

    return (weights_b, weights_d, x_add)


# -- END OF YOUR CODERUNNER SUBMISSION CODE
