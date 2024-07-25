import struct
import numpy as np
from sklearn.cluster import DBSCAN
import time

# Creazione del file binario
def create_binary_file(filename, M, N, data):
    with open(filename, 'wb') as f:
        f.write(struct.pack('i', M))  # Scrivi il numero di colonne
        f.write(struct.pack('i', N))  # Scrivi il numero di righe
        for value in data:
            f.write(struct.pack('d', value))  # Scrivi i double

# Lettura del file binario
def read_binary_file(filename):
    with open(filename, 'rb') as f:
        M = struct.unpack('i', f.read(4))[0]
        N = struct.unpack('i', f.read(4))[0]
        data = []
        for _ in range(N * M):
            data.append(struct.unpack('d', f.read(8))[0])  # Leggi i double
        matrix = np.array(data).reshape(N, M)
    return M, N, matrix

# Funzione per applicare DBSCAN e visualizzare i risultati
def apply_dbscan_and_visualize(matrix, MinPts, Eps):
    start_time = time.time()

    clustering = DBSCAN(eps=Eps, min_samples=MinPts).fit(matrix)
    labels = clustering.labels_

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"La funzione ha impiegato {execution_time} secondi per essere eseguita.")
    
    # Visualizzazione dei risultati
    #plt.scatter(matrix[:, 0], matrix[:, 1], c=labels, cmap='viridis')
    #plt.title('DBSCAN Clustering')
    #plt.xlabel('X')
    #plt.ylabel('Y')
    #plt.colorbar()
    #plt.show()
    
    return labels

# Esempio di utilizzo
M = 13  # numero di colonne
N = 100  # numero di righe
data = (np.random.rand(N * M)).astype(np.float64)  # genera dati casuali
MinPts = 3
Eps = 0.8

# Nome del file binario
nome_file_ds = f'test_{N}_{M}_64.ds'
nome_file_labels = f'test_{N}_{M}_64.labels'

# Creazione del file binario .ds
create_binary_file(nome_file_ds, M, N, data)

# Creazione del file binario .labels
create_binary_file(nome_file_labels, 1, N, data)

# Lettura del file binario
M, N, matrix = read_binary_file(nome_file_ds)

# Applicazione di DBSCAN e visualizzazione dei risultati
labels = apply_dbscan_and_visualize(matrix, MinPts, Eps)

# Stampa degli id dei cluster
labels_string = "[" + ",".join(map(str, labels)) + "]"
print(labels_string)

