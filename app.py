import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import traceback
import json
import pickle
import cv2
import pymilvus


@st.cache(allow_output_mutation=True)
def cached_cluster_model():
    pickled_kmeans_path = './data/model/bovw_cluster_model_2048.pkl'
    _cluster_alg = pickle.load(open(pickled_kmeans_path, 'rb'))
    return _cluster_alg


@st.cache(allow_output_mutation=True)
def cached_paths_base_model():
    data_path = './data/database_paths.json'
    with open(data_path, 'r') as jf:
        _base_paths_keys = json.load(jf)

    _base_paths_keys = {
        int(k): p
        for k, p in _base_paths_keys.items()
    }
    return _base_paths_keys


cluster_model = cached_cluster_model()
base_paths_keys = cached_paths_base_model()


def extract_features(_img: np.ndarray) -> np.ndarray:
    gray_image = cv2.cvtColor(_img, cv2.COLOR_RGB2GRAY)
    _k = 1000.0 / max(_img.shape[:2])
    gray_image = cv2.resize(gray_image, None, fx=_k, fy=_k, interpolation=cv2.INTER_AREA)
    _sift = cv2.SIFT_create()
    _, des = _sift.detectAndCompute(gray_image, None)
    return np.array(des)


def build_histogram(descriptor_list, cluster_alg):
    cluster_result = cluster_alg.predict(descriptor_list)
    histogram = np.histogram(
        cluster_result, bins=[
            _q
            for _q in range(
                len(cluster_alg.cluster_centers_) + 1)
        ],
        density=False
    )
    _h = np.array(histogram[0])
    return _h / np.linalg.norm(_h)


def base_search():
    st.title('Similarity search Demo')
    uploaded_file = st.file_uploader(
        'Choose an image...',
        type=['png', 'jpg', 'jpeg', 'webp']
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            image = ImageOps.exif_transpose(image)

            st.image(
                image,
                caption='Input image'
            )

            features = build_histogram(
                extract_features(np.array(image)),
                cluster_model
            )

            pymilvus.connections.connect(
                "default", host='localhost', port=19530)

            collection_name = 'SimilaritySearchSIFT'

            milvus_collection = pymilvus.Collection(collection_name)
            milvus_collection.load()

            search_params = {"metric_type": "IP", "params": {}}
            query_vectors = [features.astype(np.float32).tolist()]
            search_results = milvus_collection.search(
                query_vectors,
                'embeddings',
                param=search_params,
                limit=5,
                expr=None
            )

            indexes = search_results[0].ids
            distances = search_results[0].distances

            for idx, d in zip(indexes, distances):
                image_path = base_paths_keys[idx]
                st.image(
                    Image.open(image_path),
                    caption='Cosine distance: {:.2f}'.format(d),
                    width=540
                )
        except Exception as e:
            st.write(f"CRASHED: {traceback.format_exc()}")


if __name__ == '__main__':
    base_search()
