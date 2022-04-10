This is the code of paper "Cancer survival prediction via graph convolutional neural networks learning on protein-protein interaction networks".

you can run "train.sh" to execute the experiment, but make sure you have completed the README operation in the data directory

ran_gcn.py: details about model training\
prepare_data.py: details about data preprocessing\
layers.py & models.py: our GCN models\
nnet_survival.py: how to make loss function. Some of the content is quoted from [nnet_survival](https://github.com/MGensheimer/nnet-survival)

Model interpretability experiments are implemented with the help of [captum](https://github.com/pytorch/captum).
