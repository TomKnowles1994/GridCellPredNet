import numpy as np
import matplotlib.pyplot as plt

power_values = np.arange(500, 5100, 100)

mean_error_values = np.array([0.2200, 0.2253, 0.2129, 0.1678, 0.2492,
                              0.2214, 0.1880, 0.1920, 0.1884, 0.1761,
                              0.1938, 0.1688, 0.1428, 0.1592, 0.0925, 
                              0.0821, 0.1046, 0.0575, 0.0958, 0.0402,
                              0.0715, 0.0475, 0.0351, 0.0454, 0.0359,
                              0.0378, 0.0467, 0.1012, 0.0322, 0.0400,
                              0.0758, 0.0624, 0.0792, 0.0604, 0.0509,
                              0.0942, 0.0966, 0.1002, 0.0400, 0.0881,
                              0.0670, 0.1049, 0.0892, 0.0841, 0.0841,
                              0.1192])


fig, ax = plt.subplots(1,1)

ax.scatter(power_values, mean_error_values, c = 'DodgerBlue', label = 'Mean error')

polyfit_coefficients = np.polyfit(power_values, mean_error_values, deg = 3)

fit_values = np.polyval(polyfit_coefficients, power_values)

ax.plot(power_values, fit_values, c = 'MediumSeaGreen', label = 'Fit curve')

ax.set_xlim(0, max(power_values) + 200)
ax.set_ylim(0, max(mean_error_values) + 0.02)

ax.fill_betweenx((0,max(mean_error_values) + 0.02), 0, 400, color = 'IndianRed', alpha = 0.5, label = 'No activity')

ax.set_xlabel("Ring Current (pA)")

ax.set_ylabel("Mean Error (m)")

ax.legend()

plt.show()