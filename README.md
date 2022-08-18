Deep Anomaly Detection on Attributed Networks(SDM2019)
============

## Dominant

This is the PyTorch source code of paper "[Deep Anomaly Detection on Attributed Networks](http://www.public.asu.edu/~kding9/pdf/SDM2019_Deep.pdf)".

![The proposed framework](framework.png)

## Requirements
python==3.7.9

PyTorch=1.4.0

## Usage
```python run.py```

## Improvements
```
-> Experimented by adding three GCN layers in autoencoder
-> Trained a node-level xgboost classifier with AUC score of 0.63
```
