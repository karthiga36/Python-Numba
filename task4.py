import matplotlib.pyplot as plt
import numpy as np
from numba import njit

# Numba-optimized Pi approximation function
@njit(fastmath=True)
def approximate_pi_numba(n):
    pi_2 = 1.0
    nom, den = 2.0, 1.0
    for i in range(n):
        pi_2 *= nom / den
        if i % 2:
            nom += 2
        else:
            den += 2
    return 2 * pi_2

# Given N values
nums = [1_822_725, 22_059_421, 32_374_695, 88_754_320, 97_162_660, 200_745_654]
pi_estimations = [approximate_pi_numba(n) for n in nums]

# Correct value of Pi
true_pi = np.pi

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(nums, pi_estimations, marker='o', linestyle='-', color='b', label="Approximated Pi")
plt.axhline(y=true_pi, color='r', linestyle='--', label="True Pi (3.141592653589793)")

# Labels and title
plt.xlabel("N (Iterations)")
plt.ylabel("Pi Approximation")
plt.title("Pi Estimation vs. True Pi")
plt.legend()
plt.grid(True)

# Show plot
plt.show()
