import re
import pandas as pd
import numpy as np
import scipy.sparse as sparse
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

def to_lsa(data, n_components=None):
    N, M = data.shape[0], data.shape[1]
    if n_components is None or n_components > M:
        n_components = int(min(M / 2, estimate_n_clusters(N, 3)))
    print('LSA.n_components: ', n_components)
    svd = TruncatedSVD(n_components, random_state=0)
    normalizer = Normalizer()
    lsa = make_pipeline(svd, normalizer)
    return lsa.fit_transform(data)


def get_min_cluster_size(N):
    borders = np.array([1, 200, 300, 500, 1000, np.inf])
    # N < 200: 1, 200 <= N < 300: 2, 300 <= N < 500: 3, 500 <= N < 1000: 4, N < 1000: 5
    return np.argmax(borders>N)


def estimate_n_clusters(N, min_cluster_size, max_C = 1e3):
    ''' Оценка количества кластеров от кол-ва элементов, минимального желаемого размера кластера и максмального кол-ва кластеров.'''
    scale = max_C*min_cluster_size
    if scale:
        return min(N//2, int(max_C * N/scale /(1 + N/scale) + 1))
    return N//2


def flat_clusters_prune(labels, min_cluster_size, others):
    un_ids = sorted(zip(*[list(x) for x in np.unique(labels, return_counts=True)]), reverse=True, key=lambda x: x[1])
    mapping = {}
    for label, size in un_ids:
        if size >= min_cluster_size and label != others:
            mapping[label] = len(mapping)+1
    if not isinstance(others, str):
        others = str(others)
    digits = int(np.ceil(np.log10(len(mapping)+1)))
    mapping = {k: str(v).zfill(digits) for k, v in mapping.items()}
    return (np.vectorize(lambda x: mapping.get(x, others))(labels),
            np.vectorize(lambda x: str(x).zfill(digits), otypes=[list])(np.arange(len(mapping))+1))


class my_MiniBatchKMeans():
    ''' К-Means с добавками: вероятность, подбор количества кластеров. '''
    def __init__(self, min_cluster_size=None, max_cluster_size=0.1, others='others', n_clusters=None):
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.others = others
        self.model = None
        self.tf_idf = None
        self.centroids = None
        self.labels_ = None
        self.clusters = None

    def fit(self, tf_idf):
        self.tf_idf = tf_idf
        N = self.tf_idf.shape[0]
        if isinstance(self.n_clusters, float) and self.n_clusters < 1:
            self.n_clusters = int(self.n_clusters * N)
        elif self.n_clusters:
            self.n_clusters = min(int(self.n_clusters), 100, N//2)
        if isinstance(self.min_cluster_size, float) and self.min_cluster_size < 1:
            self.min_cluster_size = int(self.min_cluster_size * N)
        if isinstance(self.max_cluster_size, float) and self.max_cluster_size < 1:
            self.max_cluster_size = int(self.max_cluster_size * N)
        if self.min_cluster_size is None:
            self.min_cluster_size = get_min_cluster_size(N)

        if not (isinstance(self.n_clusters, int) and self.n_clusters > 1):
            self.n_clusters = (estimate_n_clusters(N, self.min_cluster_size) + estimate_n_clusters(N, self.max_cluster_size))//2

        print('Trying to divide {} items to {} clusters.'.format(self.tf_idf.shape[0], self.n_clusters))
        self.model = MiniBatchKMeans(self.n_clusters, random_state=42)
        self.model.fit(self.tf_idf)

        labels = self.model.labels_
        self.labels_, self.clusters = flat_clusters_prune(labels, self.min_cluster_size, self.others)
        has_good_clusters = isinstance(self.clusters, list) and self.clusters \
                            or isinstance(self.clusters, np.ndarray) and self.clusters.size
        if has_good_clusters:
            self.centroids = np.stack(
                [self.tf_idf[np.where(self.labels_ == l)[0], :].mean(axis=0) for l in self.clusters], axis=0)
        else:
            self.centroids = []
        return self.labels_

    def transform(self, matr=None):
        if self.tf_idf is not None:
            if matr is None:
                matr = self.tf_idf
        else:
            raise Exception("TF/IDF matrix was not passed or fit() doesn't executed!")
        return pairwise_distances(matr, self.centroids) # for non-sparce matrix distance.cdist(...)

    def predict(self, matr=None):
        return np.array([self.clusters[x] for x in np.argmin(np.asarray(self.transform(matr)), axis=1)])

    def predict_probability(self, matr=None):
        return prob_by_dist(self.transform(matr))

def group_clusters(tf_idf, labels):
    unique_clusters = np.unique(labels)
    clusters = pd.DataFrame(columns=['label', 'docs', 'centroid'])
    for cluster in unique_clusters:
        docs_ids = np.where(labels == cluster)[0]  # returns 'tuple (array,)', need 'array'
        cluster_elements = tf_idf[docs_ids]  # slice rows which form the cluster
        centroid = cluster_elements.mean(axis=0)  # make mean from rows as centroid
        centroid = np.asarray(centroid).ravel()  # np.matrix [[one line]] -> np.array [one line]
        clusters = clusters.append(
            {'label': cluster, 'docs': docs_ids, 'centroid': centroid}, ignore_index=True)
    return clusters
        
def get_top_words(tf_idf_row, words, max_n=5):
    if not isinstance(words, np.ndarray):
        words = np.array(words)
    if isinstance(tf_idf_row, np.matrix):
        row = np.asarray(tf_idf_row).ravel()  # one row matrix -> array
    else:
        row = tf_idf_row
    best_words_ids = np.argsort(row)[-max_n:][::-1]
    return words[best_words_ids]

