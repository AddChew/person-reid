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

1. Make sure that you are in the project root folder (i.e. personreid)

2. Build docker image

```shell
docker build -t personreid .
```

3. Run container based on built image

```shell
docker run -d --name personreid -p 4000:4000 personreid
```

#### Hardware specifications

| Component                        | Specs                                            |
| -------------------------------- |--------------------------------------------------|
| Docker                           | version 26.1.0 build 9714adc                     |
| Host OS                          | Linux Ubuntu 20.04.6 LTS                         |
| CPU                              | x86_64 Intel(R) Core(TM) i7-10510U CPU @ 1.80GHz |

### How to interact with the model serving endpoint

1. Spin up the model serving endpoint based on the steps in `How to spin up the model serving endpoint` if you have not done so.

2. Follow the instructions in either `Swagger UI` or `Python Client` to interact with the model serving endpoint.

#### Swagger UI

1. Navigate to http://0.0.0.0:4000/docs

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

1. Make sure that you are in the project root folder (i.e. personreid)

2. Create conda environment
```shell
conda create -n personreid-client python=3.9 -y
conda activate personreid-client
pip install -r client_requirements.txt
```

3. Open src/client.py file. You should see the following code snippet between lines 48 and 51. Set the image path in client.infer to your own image path.

```shell
if __name__ == '__main__':
    client = PersonReIDClient()
    print(client.ping()) # Test /ping
    print(client.infer("Gallery/0_1_1000.jpg")) # Set the image path to your own image path, test /infer
```

4. Run the script with the following command.

```shell
python src/client.py
```

You should see the following printed on your terminal console. The actual cluster number that you get might vary.

```shell
{'message': 'pong'}
{'cluster': 366}
```

