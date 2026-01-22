from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def build_dci(df, components):
    missing = [col for col in components if col not in df.columns]
    if missing:
        raise ValueError(f"Missing DCI components: {missing}")
    X = df[components].astype(float)
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=1, random_state=42)
    dci = pca.fit_transform(X_scaled).reshape(-1)
    dci = (dci - dci.mean()) / dci.std(ddof=0)
    return dci, pca.explained_variance_ratio_[0]
