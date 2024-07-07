import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.stats import gaussian_kde

"""
spiegazione delle colonne nel file di input GAIA
1. SOURCE_ID: Identificativo unico per ogni stella.
2. ra: Ascensione retta della stella.
3. dec: Declinazione della stella.
4. galactic_longitude: Longitudine galattica della posizione della stella.
5. galactic_latitude: Latitudine galattica della posizione della stella.
6. distance_pc: Distanza dalla stella in parsec.
7. pmra: Movimento proprio in ascensione retta.
8. pmdec: Movimento proprio in declinazione.
9. radial_velocity: Velocità radiale della stella.
10. radial_velocity_error: Errore sulla velocità radiale.
11. parallax: Parallasse della stella.
12. g_mag: Magnitudine G della stella.
13. b_r_color: Colore B-R della stella.
14. teff_gspphot: Temperatura effettiva della stella da fotometria GSP.
15. absolute_g_mag: Magnitudine assoluta G della stella.
16. x_hel_gal, y_hel_gal, z_hel_gal: Coordinate galattiche cartesiane della stella rispetto al Sole.
17. vx, vy, vz: Componenti della velocità galattica della stella rispetto al Sole.

"""



def find_clusters(npdata, eps, min_samples):
    
    # trova cluster nei dati usando l'algoritmo di DBSCAN
    
    # assicurarsi che i dati siano un arrau strutturato di NumPy
    if not isinstance(npdata, np.recarray):
        raise ValueError("Input data should be a structured numpy array.")

    # Converty l'array strutturato a un array 2D regolare
    coords = np.vstack((npdata['x_hel_gal'], npdata['y_hel_gal'], npdata['z_hel_gal'])).T

    # applica DBSCAN per trovare possibilii cluster nei dati
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = db.labels_

    # riconversione dell'array NumPy a un DataFrame di Pandas per semplificare le cose
    data = pd.DataFrame(coords, columns=['x_hel_gal', 'y_hel_gal', 'z_hel_gal'])
    data['cluster'] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print ('\n')
    print(f'Numero stimato di cluster {n_clusters}')
    print(f'Numero stimato di punti di rumore: {n_noise}')

    return data,labels



def plot_clusters(data):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # trova le labels uniche del cluster, escludendo il rumore (-1)
    unique_labels = set(data['cluster']) - {-1}
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for k, col in zip(unique_labels, colors):
        class_member_mask = (data['cluster'] == k)

        xyz = data[class_member_mask]
        ax.scatter(xyz['x_hel_gal'], xyz['y_hel_gal'], xyz['z_hel_gal'], c=[col], label=f'Cluster {k}', s=10)

    ax.set_xlabel('X Heliocentric Galactic (pc)')
    ax.set_ylabel('Y Heliocentric Galactic (pc)')
    ax.set_zlabel('Z Heliocentric Galactic (pc)')
    ax.set_title('Cluster')
    ax.legend()

    plt.show()
    


def PlotCluster(data,cldata):
    
    # Crea la mappa di tutte le stelle e evidenzia le stelle del cluster
    
    plt.figure(figsize=(10, 6))
    plt.scatter(data['ra'], data['dec'], s=1, color='skyblue', label='tutte le stelle')  # Mappa delle stelle

    # Evidenzia le stelle del cluster
    plt.scatter(cldata['ra'], cldata['dec'], s=10, color='red', label='ammasso stellare')  # Stelle del cluster

    plt.xlabel('Ascensione retta (RA)')
    plt.ylabel('Declinazione (Dec)')
    plt.title('Mappa delle Stelle con Cluster Evidenziato')
    plt.legend()
    plt.grid(True)
    plt.show()
   
    
   
def plot_stellar_map(data):    

    ###MAPPA DELLE STELLE###        

    plt.figure(figsize=(10, 6))
    plt.scatter(data['ra'], data['dec'], s=1, color='skyblue')  # 's' rappresenta la dimensione del punto

    plt.xlabel('Ascensione retta (RA)')
    plt.ylabel('Declinazione (Dec)')
    plt.title('Mappa delle Stelle')
    plt.grid(True)
    plt.show()
    


def PlotStellarDensity(data):

    ###DENSIIÀ DELLE STELLE IN PIANI DIVERSI XY, XZ, YZ###

    # Crea tre subplot per gli istogrammi
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    # Istogramma XY
    axes[0].hist2d(data['x_hel_gal'], data['y_hel_gal'], bins=50, cmap='Blues')
    axes[0].set_xlabel('X (pc)')
    axes[0].set_ylabel('Y (pc)')
    axes[0].set_title('Densità delle stelle nel piano XY')

    # Istogramma XZ
    axes[1].hist2d(data['x_hel_gal'], data['z_hel_gal'], bins=50, cmap='Greens')
    axes[1].set_xlabel('X (pc)')
    axes[1].set_ylabel('Z (pc)')
    axes[1].set_title('Densità delle stelle nel piano XZ')

    # Istogramma YZ
    axes[2].hist2d(data['y_hel_gal'], data['z_hel_gal'], bins=50, cmap='Reds')
    axes[2].set_xlabel('Y (pc)')
    axes[2].set_ylabel('Z (pc)')
    axes[2].set_title('Densità delle stelle nel piano YZ')

    # Mostra gli istogrammi
    plt.tight_layout()
    plt.show()



def PlotCMD(cldata):
    
    plt.figure(figsize=(10, 6))
    color=cldata['b_r_color']
    magnitude=cldata['g_mag']

    plt.scatter(color, magnitude, label='Data', s=0.8)


    plt.gca().invert_yaxis()  
    plt.xlabel('(B-R) color')
    plt.ylabel('Absolute G magnitude')
    plt.title('Diagramma colore-magnitudine - Cluster')
    plt.grid(True)
    plt.show()
    
    return None



def PlotProperMotions(npdata,cldata):
    
    fig=plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    pmra=npdata['pmra']
    pmdec=npdata['pmdec']

    plt.scatter(pmra, pmdec,s=0.8,alpha=0.1,color='blue', label='Tutte le Stelle')

    pmra=cldata['pmra']
    pmdec=cldata['pmdec']

    plt.scatter(pmra, pmdec,s=2,alpha=1,color='red',label='Cluster')

    ax.set_xlim(-250, 250)
    ax.set_ylim(-250, 250)
    plt.xlabel('RA moto proprio (mas/year)')
    plt.ylabel('DEC moto proprio (mas/year)')
    plt.title('Moto Prorpio')
    plt.grid(True)
    ax.legend()
    plt.show()
    
    return None



def angle_between_vectors(v1, v2):
    
    # CALCOLA L'ANGOLO FRA DUE VETTORI V1 E V2
    
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_theta = dot_product / norm_product
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))



def plot_ra_dec_rectangular(cldata):
    
    # PLOTTA RA E DEC COME COORDINATE RETTANGOLARI
    
    
    # opera una selezione sui dati per mostrare solo la zona attorno al cluster
    filtered_data = cldata[(cldata['g_mag'] < 30) & 
                          (cldata['ra'] >= 45) & (cldata['ra'] <= 85) & 
                          (cldata['dec'] >= 0) & (cldata['dec'] <= 30)]

    ra = filtered_data['ra']
    dec = filtered_data['dec']
    pmra = filtered_data['pmra']  
    pmdec = filtered_data['pmdec'] 
    magnitude = filtered_data['g_mag']
    color_index = filtered_data['b_r_color']

    ra = np.remainder(ra + 360, 360) # riduci all'intervallo 0-360

    plt.figure(figsize=(12, 10))
    
    
    # converti le magnitudini in grandezza dei simboli
    sizes = (magnitude.max() - magnitude) * 15
    
    
    # CALCOLA I MOTI PROPRI MEDI IN RA E DEC
    avg_pmra = np.mean(pmra)
    avg_pmdec = np.mean(pmdec)
    
    
    # SETTA LA SCALA DEI MOTI PROPRI PER IL PLOT
    pm_scale_ra = 0.01  
    pm_scale_dec = 0.01  

    scatter = plt.scatter(ra, dec, s=sizes, c=color_index, cmap='coolwarm', alpha=0.8, edgecolors='none')
    
    
    #CICLO SULLE STELLE e plot dei vettori (uno per stella)
    for i in range(len(ra)):
        pm_vector = np.array([pmra[i], pmdec[i]])
        avg_pm_vector = np.array([avg_pmra, avg_pmdec])
        angle_diff = angle_between_vectors(pm_vector, avg_pm_vector)
        
        # cambia colore se l'angolo fra i vettori e' maggiore o minore di 30 gradi
        if angle_diff < np.radians(15):  
            arrow_color = 'blue' 
        else:
            arrow_color = 'red'  
        
        plt.arrow(ra[i], dec[i], pmra[i] * pm_scale_ra, pmdec[i] * pm_scale_dec, color=arrow_color, 
                  width=0.02, head_width=0.1, head_length=0.1, alpha=0.5, zorder=2)
    
   
    plt.xlim(61, 71)  
    plt.ylim(10, 20)  
    
  
    plt.gca().invert_xaxis()
    
    #color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('B-R Color Index')
    
    plt.xlabel('RA (degrees)')
    plt.ylabel('DEC (degrees)')
    plt.title('RA vs DEC del Cluster (Proiezione rettangolare) con moto proprio')
    plt.grid(True)
    plt.show()





def plot_cmd(data):
    
    #PLOT DEL DIAGGRAMMA COLORE MAGNITUDINE
    plt.figure(figsize=(10, 6))

    color = data['b_r_color']
    magnitude = data['absolute_g_mag']

    plt.scatter(color, magnitude, s=1, color='blue')

    plt.gca().invert_yaxis()  # Le stelle più brillanti devono apparire più in alto nel diagramma
    plt.xlabel('(B-R) Color')
    plt.ylabel('Absolute G Magnitude')
    plt.title('Diagramma Colore-Magnitudine')
    plt.grid(True)
    plt.show()


# Funzione per il diagramma di densità
def plot_density(data, ax):
    xy = np.vstack([data['ra'], data['dec']])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = data['ra'][idx], data['dec'][idx], z[idx]
    sc = ax.scatter(x, y, c=z, s=50, edgecolor=None, cmap='viridis')
    plt.colorbar(sc, ax=ax, label='Densità')
    return sc


# Funzione per calcolare la densità media delle stelle nel campione
def calculate_stellar_density(data):
    max_distance = np.max(data['distance_pc'])
    volume = (4.0/3.0) * np.pi * max_distance**3
    stellar_density = len(data) / volume
    return max_distance, volume, stellar_density


# Funzione per calcolare la densità di stelle nel cluster
def calculate_cluster_density(cluster_data):
    X = cluster_data['x_hel_gal']
    Y = cluster_data['y_hel_gal']
    Z = cluster_data['z_hel_gal']
    
    aveX = np.average(X)
    aveY = np.average(Y)
    aveZ = np.average(Z)
    
    d = np.sqrt(aveX**2 + aveY**2 + aveZ**2)
    
    radvel = cluster_data['radial_velocity']
    AveRadVel = np.nanmean(radvel)
    
    d_star = np.sqrt((X - aveX)**2 + (Y - aveY)**2 + (Z - aveZ)**2)
    d_max = np.max(d_star)
    
    N = len(cluster_data)
    Volume = (4.0/3.0) * np.pi * d_max**3
    ClDens = float(N) / Volume
    
    return aveX, aveY, aveZ, d, AveRadVel, d_max, ClDens


# Funzione per calcolare e stampare la densità delle stelle
def plot_stellar_density(data):
    max_distance, volume, stellar_density = calculate_stellar_density(data)
    print(f"Volume del campione: {volume:.1f} pc^3")
    print(f"Densità media delle stelle: {stellar_density:.3f} stelle/pc^3")



def PlotClusterDensityMap(data):
    
   #Crea la mappa delle stelle con densità
   
   fig, ax = plt.subplots(figsize=(10, 6))
   plot_density(data, ax)
   ax.set_xlabel('Ascensione retta (RA)')
   ax.set_ylabel('Declinazione (Dec)')
   ax.set_title('Mappa delle Stelle con Densità')
   ax.grid(True)
   plt.show()



#########################################
#
# PROGRAMMA PRINCIPALE
#
#########################################

# Carica il file CSV

data = pd.read_csv('TestGaiaData.csv')

# CONVERSIONE DEL DataFrame IN UN ARRAY STRUTTURATO DI NumPy SENZA INDICI
npdata = data.to_records(index=False)


# plot mappa delle stelle
plot_stellar_map(data)

# plot densita' sui piani X, Y e Z
PlotStellarDensity(data)


# trova il cluster (applica l'algoritmo di DBSCAN)
dataclusters,labels = find_clusters(npdata, eps=4.0, min_samples=130)


# PLOTTA I CLUSTER in 3D
plot_clusters(dataclusters)


# AGGIUNGI LE LABELS ALL'ARRAY STRUTTURATO DI NumPy
dtype = npdata.dtype.descr + [('cluster', labels.dtype)]
tmp_array = np.empty(npdata.shape, dtype=dtype)

for name in npdata.dtype.names:
        tmp_array[name] = npdata[name]
        
tmp_array['cluster'] = labels

# SALVA NELL'ARRAY FINALE I DATI DEL CLUSTER
cldata=tmp_array[tmp_array['cluster']==0]



#Plot del cluster
PlotCluster(data,cldata)
    
    
# Plot del CMD di tutte le stelle
plot_cmd(data)


# Plot del CMD del cluster 
PlotCMD(cldata)

# Plot Moti Proprii
PlotProperMotions(npdata,cldata)

# Plot della posizione RA-DEC (Proiezione Rettangolare) con dimensioni basate sulla magnitudine
plot_ra_dec_rectangular(cldata)


# Crea la mappa delle stelle con densità
PlotClusterDensityMap(data)


# Calcolo della densità media delle stelle nel campione

max_distance = np.max(npdata['distance_pc'])
volume = (4.0/3.0) * np.pi * max_distance**3
stellar_density = len(npdata) / volume
print(f"\nMassima distanza nel campione: {max_distance:.1f} parsec")


# Calcola e stampa la densità delle stelle nel campione ed altri parametri
plot_stellar_density(data)

# Calcola e stampa le informazioni sul cluster
aveX, aveY, aveZ, d, AveRadVel, d_max, ClDens = calculate_cluster_density(cldata)
print('\n')
print(f"\nCoordinate del centro del cluster: Xc={aveX:.1f} pc, Yc={aveY:.1f} pc, Zc={aveZ:.1f} pc")
print(f"Distanza del cluster: {d:.1f} pc")
print(f"Velocità radiale media del centro del cluster: {AveRadVel:.1f} km/s")
print(f"Distanza massima delle stelle nel cluster: {d_max:.1f} pc")
print(f"Densità media del cluster: {ClDens:.2f} stelle/pc^3")








