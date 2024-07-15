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

3. Download the gallery images zipped file from [link](https://drive.google.com/file/d/1dGcw5C4pI331WYFo4ADHsz75HBMD0Da7/view?usp=drive_link) and place it in personreid folder. Unzip the zipped file with the command below.
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

### How to spin up the model serving endpoint

### How to interact with the model serving endpoint

1. Spin up the model serving endpoint based on the steps in `How to spin up the model serving endpoint` if you have not done so.

2. Follow the instructions in either `Swagger UI` or `Python Client` to interact with the model serving endpoint.

#### Swagger UI

1. Navigate to http://localhost:4000/docs

2. To interact with /ping endpoint, click on the GET /ping accordion and then click on the `Try it out` button on the right side of the accordion. After which, click on the `Execute` buttom at the bottom. You should see the following response.

```shell
{
  "message": "pong"
}
```

3. To interact with /infer endpoint, click on the POST /infer accordion and then click on the `Try it out` button on the right side of the accordion. After which, click on `Choose File` button at the left to upload your image file for inference. Once done, click on `Execute` button. You should see the following response. The actual cluster number that you get might vary.

```shell
{
  "cluster": 361
}
```

#### Python Client

