import pandas as pd

from som.algorithms.som import AnimalSOM

animals_data_path = 'som/data/animals.dat'
animals_name_path = 'som/data/animalnames.txt'

try:
    animals_data = pd.read_csv(animals_data_path, header=None).values.reshape(32, 84) # reshape to 32x84 matrix
except FileNotFoundError:
    print(f"File not found: {animals_data_path}")
    exit()
except Exception as e:
    print(f"An error occurred while reading the file: {e}")
    exit()

try:
    with open(animals_name_path, 'r') as f:
        animals_names = [line.strip().replace("'", "") for line in f.readlines()]
except FileNotFoundError:
    print(f"File not found: {animals_name_path}")
    exit()
except Exception as e:
    print(f"An error occurred while reading the file: {e}")
    exit()


som = AnimalSOM(grid_size=(100,), input_dim=animals_data.shape[1], learning_rate=0.2)
som.fit(animals_data, epochs=20)

pos = []
for i, animal in enumerate(animals_names):
    bmu_index = som._find_bmu(animals_data[i])
    pos.append(bmu_index[0])

sorted_animals = sorted(zip(animals_names, pos), key=lambda x: x[1])

print("Animals sorted by their position in the SOM:")
for animal, position in sorted_animals:
    print(f"{animal}: {position}")