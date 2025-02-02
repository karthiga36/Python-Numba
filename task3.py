from numba import njit
import time

# JIT compile the function using Numba
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

# List of N values
nums = [1_822_725, 22_059_421, 32_374_695, 88_754_320, 97_162_660, 200_745_654]

# Measure execution time
start_time = time.time()
results = [approximate_pi_numba(n) for n in nums]
end_time = time.time()

# Print results
print("\nNumba Optimized Results:")
for n, result in zip(nums, results):
    print(f"N={n}, Approximation: {result}")

print("Numba Execution Time:", end_time - start_time)
