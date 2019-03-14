import pandas as pd
import numpy as np
from scipy.stats import chisquare, chi2_contingency

#data = pd.read_csv('scores.csv', header=None)
data = pd.read_csv('scores_test.csv', header = None)
print(data)

# a1 = [6, 4, 5, 10]
# a2 = [8, 5, 3, 3]
# a3 = [5, 4, 8, 4]
# a4 = [4, 11, 7, 13]
# a5 = [5, 8, 7, 6]
# a6 = [7, 3, 5, 9]
# dice = np.array([a1, a2, a3, a4, a5, a6])
# print(dice)
# print("dice results", chi2_contingency(dice))

# Generally for all parties
#stats, p, dof, expected = chi2_contingency(np.array([data.iloc[:, 0]*100, data.iloc[:, 1]*100, data.iloc[:, 2]*100,
 #                                                    data.iloc[:, 3]*100, data.iloc[:, 4]*100, data.iloc[:, 5]*100]))
stats, p, dof, expected = chi2_contingency(np.array([data.iloc[:, 0]*100, data.iloc[:, 1]*100, data.iloc[:, 2]*100,
                                                     data.iloc[:, 3]*100, data.iloc[:, 4]*100]))
print("for all parties ", "stats", stats, "p-value", p, "dof", dof)

## For pairs:
for i in range(0, 5):
    for j in range(1, 5):
        p1 = data.iloc[:, i]*100
        p2 = data.iloc[:, j]*100
        stats, p, dof, expected = chi2_contingency(np.array([p1, p2]))
        # stats2, p2 = chisquare(np.array([data.iloc[:, i], data.iloc[:, j]]).T)
        print("Party 1 ", i, "Party 2 ", j, "stats", stats, "p-value", p, "degrees of freedom", dof)

"""
# Afd 0 - CDU 1
stats, p, dof, expected = chi2_contingency(np.array([data.iloc[:, 0], data.iloc[:, 1]]))
stats2, p2 = chisquare(np.array([data.iloc[:, 0], data.iloc[:, 1]]).T)
print("AFD - CDU :", "stats", stats, "p-value", p)
print("AFD - CDU :", "stats 2", stats2, "p-value 2", p2)

# Afd 0 - FDP 2
stats, p, dof, expected = chi2_contingency(np.array([data.iloc[:, 0], data.iloc[:, 2]]))
stats2, p2 = chisquare(np.array([data.iloc[:, 0], data.iloc[:, 2]]).T)
print("AFD - FDP :", "stats", stats, "p-value", p)
print("AFD - FDP :", "stats", stats2, "p-value", p2)

# Afd 0 - Grüne 3
stats, p, dof, expected = chi2_contingency(np.array([data.iloc[:, 0], data.iloc[:, 3]]))
stats2, p2 = chisquare(np.array([data.iloc[:, 0], data.iloc[:, 2]]).T)
print("AFD - Grüne :", "stats", stats, "p-value", p)

# Afd 0 - Linke 4
stats, p, dof, expected = chi2_contingency(np.array([data.iloc[:, 0], data.iloc[:, 4]]))
stats2, p2 = chisquare(np.array([data.iloc[:, 0], data.iloc[:, 2]]).T)
print("AFD - Linke :", "stats", stats, "p-value", p)

# Afd 0 - SPD 5
stats, p, dof, expected = chi2_contingency(np.array([data.iloc[:, 0], data.iloc[:, 5]]))
stats2, p2 = chisquare(np.array([data.iloc[:, 0], data.iloc[:, 2]]).T)
print("AFD - SPD :", "stats", stats, "p-value", p)

# CDU 1 - FDP 2
stats, p, dof, expected = chi2_contingency(np.array([data.iloc[:, 1], data.iloc[:, 2]]))
stats2, p2 = chisquare(np.array([data.iloc[:, 0], data.iloc[:, 2]]).T)
print("CDU - FDP :", "stats", stats, "p-value", p)

# CDU 1 - Grüne 3
stats, p, dof, expected = chi2_contingency(np.array([data.iloc[:, 1], data.iloc[:, 3]]))
stats2, p2 = chisquare(np.array([data.iloc[:, 0], data.iloc[:, 2]]).T)
print("CDU - Grüne :", "stats", stats, "p-value", p)

# CDU 1 - Linke 4
stats, p, dof, expected = chi2_contingency(np.array([data.iloc[:, 1], data.iloc[:, 4]]))
stats2, p2 = chisquare(np.array([data.iloc[:, 0], data.iloc[:, 2]]).T)
print("CDU - Linke :", "stats", stats, "p-value", p)

# CDU 1 - SPD 5
stats, p, dof, expected = chi2_contingency(np.array([data.iloc[:, 1], data.iloc[:, 5]]))
stats2, p2 = chisquare(np.array([data.iloc[:, 0], data.iloc[:, 2]]).T)
print("CDU - SPD :", "stats", stats, "p-value", p)

# FDP 2 - Grüne 3
stats, p, dof, expected = chi2_contingency(np.array([data.iloc[:, 2], data.iloc[:, 3]]))
stats2, p2 = chisquare(np.array([data.iloc[:, 0], data.iloc[:, 2]]).T)
print("FDP - Grüne :", "stats", stats, "p-value", p)

# FDP 2 - Linke 4
stats, p, dof, expected = chi2_contingency(np.array([data.iloc[:, 2], data.iloc[:, 4]]))
stats2, p2 = chisquare(np.array([data.iloc[:, 0], data.iloc[:, 2]]).T)
print("FDP - Linke :", "stats", stats, "p-value", p)

# FDP 2 - SPD 5
stats, p, dof, expected = chi2_contingency(np.array([data.iloc[:, 2], data.iloc[:, 5]]))
print("AFD - SPD :", "stats", stats, "p-value", p)

# Grüne 3 - Linke 4
stats, p, dof, expected = chi2_contingency(np.array([data.iloc[:, 3], data.iloc[:, 4]]))
print("Grüne - Linke :", "stats", stats, "p-value", p)

# Grüne 3 - SPD 5
stats, p, dof, expected = chi2_contingency(np.array([data.iloc[:, 3], data.iloc[:, 5]]))
print("Grüne - SPD :", "stats", stats, "p-value", p)

# Linke 4 - SPD 5
stats, p, dof, expected = chi2_contingency(np.array([data.iloc[:, 4], data.iloc[:, 5]]))
print("Linke - SPD :", "stats", stats, "p-value", p)
"""

