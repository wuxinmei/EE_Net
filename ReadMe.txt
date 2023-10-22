# A PyTorch implementation of EE_Net on MPIIFaceGaze dataset


## Requirements

* Linux (Tested on Ubuntu only)
* Python >= 3.7

```bash
pip install -r requirements.txt
```


## Download the dataset and preprocess it

```

### MPIIFaceGaze

```bash
bash scripts/download_mpiifacegaze_dataset.sh
python tools/preprocess_mpiifacegaze.py --dataset datasets/MPIIFaceGaze_normalized -o datasets/
```

### Training and Evaluation

By running the following code, you can train a model using all the
data except the person with ID 0, and run test on that person.

```bash
python train.py --config configs/mpiifacegaze/EE_Net_train.yaml
python evaluate.py --config configs/mpiigaze/EE_Net_eval.yaml
```


## Results

### MPIIFaceGaze

| Model     | Mean Test Angle Error [degree]
|:----------|:------------------------------:|
| EE_Net_v0   |             2.86              |
| EE_Net_v4 |              2.76              |  


