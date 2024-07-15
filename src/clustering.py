import os
import umap
import torch

import hdbscan
import logging

import numpy as np

from feature_extractor import FeatureExtractor


random_state = 0
np.random.seed(random_state)
logging.basicConfig(level = logging.INFO, format = "%(asctime)s %(levelname)s %(module)s:%(lineno)d - %(message)s")


root = "../../"
gallery_dir = os.path.join(root, "Gallery")
files = sorted(os.listdir(gallery_dir ))
filepaths = [os.path.join(gallery_dir , path) for path in files]


logging.info("Load pretrained feature extractor")
model_path = os.path.join(root, "models/osnet_x0_25_market_256x128_amsgrad_ep180_stp80_lr0.003_b128_fb10_softmax_labelsmooth_flip.pth")
extractor = FeatureExtractor(
    model_name = 'osnet_x0_25',
    model_path = model_path,
    device = 'cpu',
)


logging.info("Extract embeddings")
with torch.no_grad():
    embeddings = extractor(files).numpy()
logging.info(f"Embeddings shape: {embeddings.shape}")


embeddings_path = os.path.join(root, "embeddings.joblib")
logging.info(f"Save embeddings to {embeddings_path}")
np.save(embeddings, embeddings_path)


logging.info("Run UMAP algorithm")
umap_model = umap.UMAP(n_neighbors = 15, n_components = 2, metric = "cosine", random_state = random_state)
umap_embeddings = umap_model.fit(embeddings).embedding_
logging.info(f"UMAP embeddings shape: {umap_embeddings.shape}")


logging.info("Run HDBSCAN algorithm")
hdbscan_model = hdbscan.HDBSCAN(min_cluster_size = 11, min_samples = 5)
hdbscan_model.fit(umap_embeddings)



logging.info(f"UMAP embeddings shape: {umap_embeddings.shape}") # TODO: log number of clusters and clusters