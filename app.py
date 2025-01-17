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

# ✅ Chargement du modèle CLARA existant
@st.cache_data
def load_model(filename="model_clara.pkl"):
    with open(filename, 'rb') as file:
        model = dill.load(file)
    return model

# ✅ Chargement et préparation des données avec gestion d'erreurs
@st.cache_data
def load_data(file):
    try:
        data = pd.read_csv(file, encoding='utf-8', sep=',')
        if data.empty:
            st.error("❌ Le fichier est vide ou mal formaté.")
            return pd.DataFrame()

        data.drop(columns=['CUST_ID'], inplace=True, errors='ignore')
        data.fillna(data.median(), inplace=True)
        numeric_data = data.select_dtypes(include='number')
        normalized_data = (numeric_data - numeric_data.mean()) / numeric_data.std()
        
        if numeric_data.empty:
            st.error("❌ Aucune colonne numérique détectée.")
            return pd.DataFrame()

        return normalized_data,numeric_data
    
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du fichier : {e}")
        return pd.DataFrame()

# ✅ Génération de labels synthétiques avec KMeans
def generate_synthetic_labels(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(data)

# ✅ Évaluation du clustering
def evaluate_clustering(data, labels, synthetic_labels):
    silhouette = silhouette_score(data, labels)
    db_index = davies_bouldin_score(data, labels)
    ch_score = calinski_harabasz_score(data, labels)

    # Calcul des métriques supervisées
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

# ✅ Visualisation des clusters
def plot_clusters(data, labels, medoids):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    cluster_centers = pca.transform(medoids)

    plt.figure(figsize=(10, 6))

    unique_clusters = sorted(set(labels))
    palette = sns.color_palette("hsv", len(unique_clusters))

    # Tracer chaque cluster avec des couleurs distinctes et numéro de cluster
    for cluster, color in zip(unique_clusters, palette):
        plt.scatter(
            reduced_data[labels == cluster, 0], 
            reduced_data[labels == cluster, 1], 
            label=f'Cluster {cluster}', 
            alpha=0.7, 
            color=color, 
            edgecolor='k', 
            s=40
        )
        
    # Affichage des médoïdes en rouge
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], 
                color='red', marker='X', s=200, label='Médoïdes')

    plt.title("Visualisation des clusters avec CLARA")
    plt.xlabel("Composante Principale 1")
    plt.ylabel("Composante Principale 2")
    plt.legend(title="Clusters", loc="best")
    plt.grid(True)
    st.pyplot(plt)


#Outliers
def plot_outliers(original_data):
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=original_data, orient="h", palette="Set2")
    plt.title("Detection des Outliers dans les Variables")
    plt.xlabel("Valeurs")
    plt.ylabel("Variables")
    plt.grid(True) 
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

# ✅ Interface Streamlit
def main():
    st.sidebar.title("Sections")
    section = st.sidebar.radio("Navigation", ["Accueil", "Traitement des Données", "Visualisation de l'algorithme", "Métriques d'Évaluation"])

    # Initialisation de la session
    if 'uploaded_file' not in st.session_state:
        st.session_state['uploaded_file'] = None
        st.session_state['data_loaded'] = False

    # ✅ Section Accueil
    if section == "Accueil":
        st.title("Algorithme CLARA avec Traitement Parallèle")
        st.write("""
        L'algorithme **CLARA (Clustering Large Applications)** est une version optimisée de l'algorithme **PAM (Partitioning Around Medoids)** 
        qui permet de traiter de grands ensembles de données. Il utilise des **sous-échantillons** et répartit le calcul sur plusieurs 
        **processus parallèles** pour identifier les **médoïdes** avec un coût minimal.
        """)
        st.image("CLARA_processus_fonctionnement.png", caption="Fonctionnement de l'Algorithme CLARA", use_container_width=True)

    # ✅ Section Importation des Données
    elif section == "Traitement des Données":
        st.title("📂 Importation des Données")
        uploaded_file = st.file_uploader("Importer un fichier CSV", type=["csv"])
        if uploaded_file is not None:
            data,original_data = load_data(uploaded_file)
            if not data.empty:
                st.session_state['uploaded_file'] = uploaded_file
                st.session_state['data'] = data
                st.session_state['data_loaded'] = True
                st.write("### Aperçu des données :")
                st.dataframe(original_data.head())
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
                st.dataframe(original_data.describe())
                st.write("### Visualisation des Outliers:")
                plot_outliers(original_data)
                
            else:
                st.warning("⚠️ Aucune donnée valide trouvée.")

    # ✅ Section Visualisation de l'algorithme
    elif section == "Visualisation de l'algorithme":
        st.title("📊 Visualisation des Clusters")
        if st.session_state.get('data_loaded', False):
            data = st.session_state['data']
            model = load_model()
            start_time = time.time()
            labels_clara = model.fit_predict(data)
            end_time = time.time()
            temps_execution = end_time - start_time
            plot_clusters(data, labels_clara, data.iloc[model.medoid_indices_])
            st.success(f"⏱️ Temps d'exécution du CLARA : {temps_execution:.2f} secondes")  
        else:
            st.warning("⚠️ Veuillez importer les données avant de passer à la visualisation.")

    # ✅ Métriques d'Évaluation
    elif section == "Métriques d'Évaluation":
        st.title("📈 Métriques d'Évaluation")
        if st.session_state.get('data_loaded', False):
            data = st.session_state['data']
            model = load_model()
            labels = model.fit_predict(data)
            synthetic_labels = generate_synthetic_labels(data, n_clusters=len(set(labels)))
            metrics = evaluate_clustering(data, labels, synthetic_labels)
            st.write("### Métriques d'évaluation :")
            for key, value in metrics.items():
                st.write(f"- **{key}** : {value:.4f}")
        else:
            st.warning("⚠️ Veuillez importer les données avant d'afficher les métriques.")

if __name__ == "__main__":
    main()
