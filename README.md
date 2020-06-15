# Pretrained-Meta-GNN
A (failed) attempt to combine pretraining and MAML on GNN

核心代码为MAML.py(只有bio有，为近似版MAML) true-MAML.py MAML-main.py my_fake_model.py
其余代码大多为[pretrained-gnns](https://github.com/snap-stanford/pretrain-gnns) 原始代码
数据集需要自行下载[chem data](http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip) (2.5GB) [bio data](http://snap.stanford.edu/gnn-pretrain/data/bio_dataset.zip) (2GB)

建议阅读chem的代码（因为是后写的，代码整体比较简洁）
