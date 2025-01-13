import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import dill
import time

# Chargement du modèle CLARA existant
@st.cache_data
def load_model(filename="model_clara.pkl"):
    with open(filename, 'rb') as file:
        model = dill.load(file)
    return model

# Chargement et préparation des données
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    data.drop(columns=['CUST_ID'], inplace=True, errors='ignore')
    data.fillna(data.median(), inplace=True)
    numeric_data = data.select_dtypes(include='number')
    return (numeric_data - numeric_data.mean()) / numeric_data.std()

# Génération de labels synthétiques avec KMeans
def generate_synthetic_labels(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(data)

# Évaluation du clustering
def evaluate_clustering(data, labels, synthetic_labels):
    silhouette = silhouette_score(data, labels)
    db_index = davies_bouldin_score(data, labels)
    ch_score = calinski_harabasz_score(data, labels)
    
    # Calcul des métriques supervisées avec labels synthétiques
    ari = adjusted_rand_score(synthetic_labels, labels)
    nmi = normalized_mutual_info_score(synthetic_labels, labels)

    metrics = {
        'Silhouette Coefficient': silhouette,
        'Davies-Bouldin Index': db_index,
        'Calinski-Harabasz Index': ch_score,
        'Adjusted Rand Index (ARI)': ari,
        'Normalized Mutual Information (NMI)': nmi
    }
    
    return metrics

# Visualisation des clusters
def plot_clusters(data, labels, medoids):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    cluster_centers = pca.transform(medoids)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], color='red', marker='x', s=200, label='Médoïdes')
    plt.title("Visualisation des clusters avec CLARA")
    plt.xlabel("Composante Principale 1")
    plt.ylabel("Composante Principale 2")
    plt.legend()
    st.pyplot(plt)

# Interface Streamlit
def main():
    st.sidebar.title("Sections")
    section = st.sidebar.radio('Navigation',["Accueil", "Traitement des Données", "Visualisation de l'algorithme", "Métriques d'Évaluation"])

    st.session_state.setdefault('uploaded_file', None)

    if section == "Accueil":
        st.title("Algorithme CLARA avec Traitement Parallèle utilisant des Processus")
        st.write("""
        L'algorithme **CLARA (Clustering Large Applications)** est une version optimisée de l'algorithme **PAM (Partitioning Around Medoids)** 
        qui permet de traiter de grands ensembles de données. Il utilise des **sous-échantillons** et répartit le calcul sur plusieurs 
        **processus parallèles** pour identifier les **médoïdes** (centres de clusters) avec un coût minimal.
        """)
        st.image("CLARA_processus_fonctionnement.png", caption="Fonctionnement de l'Algorithme CLARA avec Traitement Parallèle", use_container_width=True)

    elif section == "Traitement des Données":
        st.title("📂 Importation des Données")
        if st.session_state['uploaded_file'] is None:
            st.session_state['uploaded_file'] = st.file_uploader("Importer un fichier CSV", type=["csv"])
        if st.session_state['uploaded_file'] is not None:
            data = load_data(st.session_state['uploaded_file'])
            st.write("### Aperçu des données :")
            st.dataframe(data.head())
            st.markdown("""
### **Informations sur les Variables :**

- **`BALANCE`** : Montant du solde restant sur le compte pour effectuer des achats.  
- **`BALANCE_FREQUENCY`** : Fréquence de mise à jour du solde (score entre 0 et 1).  
  - *1 = Solde fréquemment mis à jour*  
  - *0 = Rarement mis à jour*  
- **`PURCHASES`** : Montant total des achats effectués à partir du compte.  
- **`ONEOFF_PURCHASES`** : Montant maximum des achats effectués en une seule fois.  
- **`INSTALLMENTS_PURCHASES`** : Montant total des achats réglés en plusieurs fois.  
- **`CASH_ADVANCE`** : Montant des avances de fonds versées par l'utilisateur.  
- **`PURCHASES_FREQUENCY`** : Fréquence des achats effectués (score entre 0 et 1).  
  - *1 = Achats fréquents*  
  - *0 = Achats peu fréquents*  
- **`ONEOFFPURCHASESFREQUENCY`** : Fréquence des achats ponctuels (score entre 0 et 1).  
- **`PURCHASESINSTALLMENTSFREQUENCY`** : Fréquence des achats en plusieurs fois (score entre 0 et 1).  
- **`CASHADVANCEFREQUENCY`** : Fréquence des avances de fonds en espèces.  
- **`CASHADVANCETRX`** : Nombre de transactions effectuées avec des avances de fonds.  
- **`PURCHASES_TRX`** : Nombre total de transactions d'achat.  
- **`CREDIT_LIMIT`** : Limite de crédit accordée à l'utilisateur.  
- **`PAYMENTS`** : Montant total des paiements effectués par l'utilisateur.  
- **`MINIMUM_PAYMENTS`** : Montant minimum des paiements effectués.  
- **`PRCFULLPAYMENT`** : Pourcentage des paiements effectués en totalité.  
- **`TENURE`** : Durée d'utilisation de la carte de crédit (en mois).  
""")
            st.write(f"**Nombre de lignes :** {data.shape[0]}")
            st.write(f"**Nombre de colonnes :** {data.shape[1]}")
            st.write("### Statistiques descriptives :")
            st.dataframe(data.describe())

    elif section == "Visualisation de l'algorithme":
        st.title("📊 Visualisation des Clusters")
        if st.session_state['uploaded_file'] is not None:
            data = load_data(st.session_state['uploaded_file'])
            model = load_model()

            # CLARA Clustering            
            start_time = time.time()
            labels_clara = model.fit_predict(data)
            end_time = time.time()
            temps_execution = end_time - start_time
            st.subheader("Projection des clusters CLARA en 2D")
            plot_clusters(data, labels_clara, data.iloc[model.medoid_indices_])
            
            # ✅ Display execution time
            st.success(f"⏱️ Temps d'execution du CLARA Clustering : {temps_execution:.2f} secondes")
            
        else:
            st.warning("⚠️ Veuillez importer un fichier dans la section Données.")

    elif section == "Métriques d'Évaluation":
        st.title("📈 Métriques d'Évaluation")
        if st.session_state['uploaded_file'] is not None:
            data = load_data(st.session_state['uploaded_file'])
            model = load_model()
            labels = model.fit_predict(data)

            # Génération de labels synthétiques avec KMeans
            synthetic_labels = generate_synthetic_labels(data, n_clusters=len(set(labels)))

            # Évaluation complète
            metrics = evaluate_clustering(data, labels, synthetic_labels)
            
            st.write("### Métriques d'évaluation :")
            for key, value in metrics.items():
                st.write(f"- **{key}** : {value:.4f}")

        else:
            st.warning("⚠️ Veuillez importer un fichier dans la section Données.")

if __name__ == "__main__":
    main()
