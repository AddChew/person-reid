import os
import umap
import torch

import hdbscan
import logging
import numpy as np
import pandas as pd

from utils import FeatureExtractor


random_state = 0
np.random.seed(random_state)
logging.basicConfig(level = logging.INFO, format = "%(asctime)s %(levelname)s %(module)s:%(lineno)d - %(message)s")


root = "./"
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
    embeddings = extractor(filepaths).numpy()
logging.info(f"Embeddings shape: {embeddings.shape}")


embeddings_path = os.path.join(root, "outputs", "embeddings.npy")
logging.info(f"Save embeddings to {embeddings_path}")
np.save(embeddings_path, embeddings)


logging.info("Run UMAP algorithm")
umap_model = umap.UMAP(n_neighbors = 15, n_components = 2, metric = "cosine", random_state = random_state)
umap_embeddings = umap_model.fit(embeddings).embedding_
logging.info(f"UMAP embeddings shape: {umap_embeddings.shape}")


logging.info("Run HDBSCAN algorithm")
hdbscan_model = hdbscan.HDBSCAN(min_cluster_size = 11, min_samples = 5)
hdbscan_model.fit(umap_embeddings)
logging.info(f"Labels {hdbscan_model.labels_}")
logging.info(f"No of unique labels: {hdbscan_model.labels_.max() + 2}")


labels = pd.DataFrame({"Filename": files, "ClusterID": hdbscan_model.labels_})
logging.info(labels.head())
logging.info(f"Labels shape: {labels.shape}")


csv_path = os.path.join(root, "outputs", "Addison_clusterid.csv")
logging.info(f"Save labels to CSV file {csv_path}")
labels.to_csv(csv_path, index = False)