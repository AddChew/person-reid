# Person Re-identification

## Instructions

The model serving endpoint depends on the clustering results generated from the clustering algorithm. You can either use the provided clustering results in outputs folder or choose to run the clustering algorithm to generate our own clustering results. Refer to `How to run the clustering algorithm` section if you want to run the clustering algorithm. Otherwise you can proceed to `How to spin up the model serving endpoint` section.

### How to run the clustering algorithm

1. Git clone this repository
```shell
git clone https://github.com/AddChew/person-reid.git
```

2. Navigate into personreid folder
```shell
cd personreid
```

3. Download the gallery images zipped file from [link](https://drive.google.com/file/d/1dGcw5C4pI331WYFo4ADHsz75HBMD0Da7/view?usp=drive_li
nk) and place it in personreid folder. Unzip the zipped file with the command below.
```shell
unzip Gallery.zip
```

4. Create conda environment
```shell
conda create -n personreid-clustering python=3.9 -y
conda activate personreid-clustering
pip install -r clustering_requirements.txt
```

5. Run the clustering algorithm, which will produce 2 artifacts in outputs folder (i.e. embeddings.npy, Addison_clusterid.csv)
```shell
python src/clustering.py
```