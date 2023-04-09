
# Learning to Retain while Acquiring: Combating Distribution-Shift in Adversarial Data-Free Knowledge Distillation. CVPR 2023 [![arXiv](https://img.shields.io/badge/arXiv-2302.14290-b31b1b.svg)](https://arxiv.org/abs/2302.14290)

🔥 **Accepted at CVPR 2023!**

## Introduction
Data-free Knowledge Distillation (DFKD) has gained popularity recently, with the fundamental idea of carrying out knowledge transfer from a Teacher neural network to a Student neural network in the absence of training data. However, in the Adversarial DFKD framework, the student network's accuracy, suffers due to the non-stationary distribution of the pseudo-samples under multiple generator updates. To this end, at every generator update, we aim to maintain the student's performance on previously encountered examples while acquiring knowledge from samples of the current distribution. Thus, we propose a meta-learning inspired framework by treating the task of Knowledge-Acquisition (learning from newly generated samples) and Knowledge-Retention (retaining knowledge on previously met samples) as meta-train and meta-test, respectively. Hence, we dub our method as Learning to Retain while Acquiring. Moreover, we identify an implicit aligning factor between the Knowledge-Retention and Knowledge-Acquisition tasks indicating that the proposed student update strategy enforces a common gradient direction for both tasks, alleviating interference between the two objectives. Finally, we support our hypothesis by exhibiting extensive evaluation and comparison of our method with prior arts on multiple datasets.

## Instructions to run the code:

Create environment using `virtualenv` (We ran our experiments on Python 3.8).

```bash
virtualenv -p python3 lra
source lra/bin/activate
pip3 install -r requirements.txt
cd src
```
Store all the pre-trained teacher model in `/cache/models`. [Link](https://purdue0-my.sharepoint.com/:u:/g/personal/pate1332_purdue_edu/Ee5BueIz7L9PvpBSTNef3joB8nXIujnxya1wnXpDTW9ssg?e=tYh93J)

Experiments with their hyperparameters are present in the `./src/configs` directory as `.json` files.

Use the following command to run an experiment:
```bash
python main.py ./configs/Ours-2.json
```
To run a different experiment change the `.json` file.

# Reference
If you find our code useful for your research, please cite our paper.
```
@inproceedings{
patel2023learning,
title={Learning to Retain while Acquiring: Combating Distribution-Shift in Adversarial Data-Free Knowledge Distillation},
author={Patel, Gaurav and Mopuri, Konda Reddy and Qiu, Qiang},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
year={2023}
}
```