import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from som.algorithms.som import TourSOM

cities_path = 'som/data/cities.dat'

try:
    # comment='%' tells Pandas to ignore the header text
    # sep=r'[;,]' with engine='python' handles both the comma separator and the trailing semicolon
    raw_data = pd.read_csv(cities_path, header=None, comment='%', sep=r'[;,]', engine='python')
    
    # We only take the first two columns (index 0 and 1) because the semicolon 
    # might cause Pandas to create a third empty column of NaNs
    cities_data = raw_data.iloc[:, :2].values.astype(float)
    
except Exception as e:
    print(f"An error occurred while reading the file: {e}")
    exit()

som = TourSOM(grid_size=(10,), input_dim=2, learning_rate=0.2)
som.fit(cities_data, epochs=100)

plt.figure(figsize=(8, 6))

plt.scatter(cities_data[:, 0], cities_data[:, 1], c='red', s=100, zorder=5, label='Cities')

# To visually close the circle, we append the first node to the end of the weights array
tour_coordinates = np.vstack((som.weights, som.weights[0]))

plt.plot(tour_coordinates[:, 0], tour_coordinates[:, 1], c='blue', marker='o', 
         linestyle='-', linewidth=2, markersize=6, label='SOM Tour')

plt.title('SOM Cyclic Tour (Travelling Salesman)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

results_dir = 'som/results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

fig_path = os.path.join(results_dir, 'tsp_graph.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to {fig_path}")

plt.show()