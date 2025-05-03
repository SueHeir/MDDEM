import numpy as np
import matplotlib.pyplot as plt

# Read data from GranularTemp.txt
data = np.loadtxt('GranularTemp.txt')



new_data = np.log10(data / data[0])  # Normalize the data

time = np.arange(1, len(new_data)+1)  # Create a time array
new_time = np.log10(time / time[0])  # Normalize the time


# Plot the data
plt.plot(new_time, new_data, label='Granular Temperature', color='blue')

# Plot the reference line of slope -2
x = np.linspace(0, 10, 100)
y = -2 * x + 4.4
plt.plot(x, y, label='Slope -2', color='red')
plt.legend()
plt.xscale('linear')
plt.yscale('linear')
plt.xlabel('Log_10 (t/t0)')
plt.ylabel('Log_10 (T/T_0)')
plt.ylim(-4,0.05)
plt.xlim(0,5.0)
plt.savefig('GranularTemp.png', dpi=300)