# AirGNN

This repository includes the official implementation of AirGNN in the paper **"Graph Neural Networks with Adaptive Residual"** [NeurIPS 2021]. 

[Xiaorui Liu](http://cse.msu.edu/~xiaorui/), [Jiayuan Ding](https://scholar.google.com/citations?user=7lwkXGEAAAAJ&hl=en), [Wei Jin](http://cse.msu.edu/~jinwei2/), [Han Xu](https://cse.msu.edu/~xuhan1/), [Yao Ma](http://cse.msu.edu/~mayao4/), [Zitao Liu](http://www.zitaoliu.com/), [Jiliang Tang](http://www.cse.msu.edu/~tangjili/). [**Graph Neural Networks with Adaptive Residual**](https://openreview.net/pdf?id=hfkER_KJiNw).  

Related materials: [paper](https://openreview.net/pdf?id=hfkER_KJiNw), [appendix](https://openreview.net/attachment?id=hfkER_KJiNw&name=supplementary_material), [slide](https://cse.msu.edu/~xiaorui/files/Slide_AirGNN.pdf), [poster](https://cse.msu.edu/~xiaorui/files/Poster_AirGNN.pdf)


![](https://raw.githubusercontent.com/lxiaorui/AirGNN/master/AMP.png)

## Code Description

The model works well under both poison setting (attacks happen before training) and evasion setting (attacks happen after training). It also works under multiple attacking methods since the development of AirGNN does not reply on how the features are attacked.

- model.py: definitation of the AirGNN model

- train_model: training and saving the model (need to tune lambda)

- adv_attack: generating adversarial feature attacks

- test_adv: test the performance of saved model on the generated adversarial datasets

Example: 
```
 python train_model.py  --dataset Cora --runs 10 --model AirGNN --dropout 0.8 --lr 0.01 --lambda_amp 0.5
```


## Reference
Please cite our paper if you find the paper or code to be useful. Thank you!

```
@inproceedings{
liu2021graph,
title={Graph Neural Networks with Adaptive Residual},
author={Xiaorui Liu and Jiayuan Ding and Wei Jin and Han Xu and Yao Ma and Zitao Liu and Jiliang Tang},
booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
year={2021},
url={https://openreview.net/forum?id=hfkER_KJiNw}
}
```
