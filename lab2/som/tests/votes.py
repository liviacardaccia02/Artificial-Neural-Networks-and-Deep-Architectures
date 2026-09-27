import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from som.algorithms.som import VotesSOM

votes_path = 'som/data/votes.dat'
district_path = 'som/data/mpdistrict.dat'
names_path = 'som/data/mpnames.txt'
party_path = 'som/data/mpparty.dat'
gender_path = 'som/data/mpsex.dat'    


try:
    votes_data = pd.read_csv(votes_path, header=None).values.reshape(349, 31) # reshape to 349x31 matrix
    # Load labels and flatten to 1D arrays (length 349)
    party_data = pd.read_csv(party_path, header=None, comment='%', sep=r'[;,]', engine='python').values.flatten()
    gender_data = pd.read_csv(gender_path, header=None, comment='%', sep=r'[;,]', engine='python').values.flatten()
    district_data = pd.read_csv(district_path, header=None, comment='%', sep=r'[;,]', engine='python').values.flatten()
except Exception as e:
    print(f"An error occurred while reading the data: {e}")
    exit()

som = VotesSOM(grid_size=(10,10), input_dim=votes_data.shape[1], learning_rate=0.2)
som.fit(votes_data, epochs=200)

bmu_positions = np.array([som._find_bmu(x) for x in votes_data])

# jitter needed for the scatter plot to avoid overlapping points
jitter = np.random.uniform(-0.3, 0.3, size=bmu_positions.shape)
bmu_positions_jittered = bmu_positions + jitter

party_mapping = {
    0: ('No Party', 'gray'),
    1: ('M (Moderates)', 'blue'),
    2: ('FP (Liberals)', 'cornflowerblue'),
    3: ('S (Social Democrats)', 'red'),
    4: ('V (Left Party)', 'darkred'),
    5: ('MP (Green Party)', 'green'),
    6: ('KD (Christian Dems)', 'darkblue'),
    7: ('C (Centre Party)', 'orange')
}

gender_mapping = {
    0: ('Male', 'green'),
    1: ('Female', 'orange')
}

results_dir = 'som/results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Plot by party
plt.figure(figsize=(10, 8))
for party_id, (party_name, party_color) in party_mapping.items():
    mask = (party_data == party_id)
    plt.scatter(bmu_positions_jittered[mask, 1], bmu_positions_jittered[mask, 0], 
                c=party_color, label=party_name, s=40, alpha=0.8, edgecolors='white', linewidth=0.5)

plt.title('SOM Topological Clustering (By Party)')
plt.xlim(-1, 10); plt.ylim(-1, 10); plt.gca().invert_yaxis()
plt.xticks(np.arange(-0.5, 10.5, 1), labels=[]); plt.yticks(np.arange(-0.5, 10.5, 1), labels=[])
plt.grid(True, linestyle='-', color='black', alpha=0.15)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Political Parties")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'parties_graph.png'), dpi=300, bbox_inches='tight')
plt.show()

# Plot by gender
plt.figure(figsize=(10, 8))
for gender_id, (gender_name, gender_color) in gender_mapping.items():
    mask = (gender_data == gender_id)
    plt.scatter(bmu_positions_jittered[mask, 1], bmu_positions_jittered[mask, 0], 
                c=gender_color, label=gender_name, s=40, alpha=0.8, edgecolors='white', linewidth=0.5)

plt.title('SOM Topological Clustering (By Gender)')
plt.xlim(-1, 10); plt.ylim(-1, 10); plt.gca().invert_yaxis()
plt.xticks(np.arange(-0.5, 10.5, 1), labels=[]); plt.yticks(np.arange(-0.5, 10.5, 1), labels=[])
plt.grid(True, linestyle='-', color='black', alpha=0.15)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Gender")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'gender_graph.png'), dpi=300, bbox_inches='tight')
plt.show()

# Plot by district
plt.figure(figsize=(10, 8))
scatter = plt.scatter(bmu_positions_jittered[:, 1], bmu_positions_jittered[:, 0], 
                      c=district_data, cmap='tab20', s=40, alpha=0.8, edgecolors='white', linewidth=0.5)

plt.title('SOM Topological Clustering (By District)')
plt.xlim(-1, 10); plt.ylim(-1, 10); plt.gca().invert_yaxis()
plt.xticks(np.arange(-0.5, 10.5, 1), labels=[]); plt.yticks(np.arange(-0.5, 10.5, 1), labels=[])
plt.grid(True, linestyle='-', color='black', alpha=0.15)
plt.colorbar(scatter, label='District ID (1 - 29)')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'district_graph.png'), dpi=300, bbox_inches='tight')
plt.show()