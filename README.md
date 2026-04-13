
# Cross-Space Synergy for Multimodal Emotion Recognition in Conversation

This repository contains the official implementation of our paper [Cross-Space Synergy: A Unified Framework for Multimodal Emotion Recognition in Conversation](https://ojs.aaai.org/index.php/AAAI/article/view/39602).

## Requirements

The required Python packages are listed in `requirements.txt`. Please install them before running the code:

```bash
pip install -r requirements.txt
```

## Dataset

The preprocessed multimodal features used in this work can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1TwT9z6N6SJadsVkDhSNVBiF9ZygEyA6l).

After downloading, please place the files in the `data/` directory:

```text
data/iemocap_multimodal_features.pkl
data/meld_multimodal_features.pkl
```
The features are derived from [SDT](https://github.com/butterfliesss/SDT?tab=readme-ov-file). We thank the authors of SDT for making this line of research possible.



## Run

You can train the model by running the corresponding training script:

```bash
python train_IEMOCAP.py
python train_MELD.py
```

## Project Structure

```text
.
├── dataloader.py
├── model.py
├── train_IEMOCAP.py
├── requirements.txt
└── data/
    ├── iemocap_multimodal_features.pkl
```

## Citation
If you find this work helpful to your research, please consider citing our paper. Thank you!
```text
@article{CSS_2026,
  title={Cross-Space Synergy: A Unified Framework for Multimodal Emotion Recognition in Conversation},
  volume={40},
  DOI={10.1609/aaai.v40i29.39602},
  number={29},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  author={Lyu, Xiaosen and Xiong, Jiayu and Chen, Yuren and Wang, Wanlong and Dai, Xiaoqing and Wang, Jing},
  year={2026},
  month={Mar.},
  pages={24226-24234} }
```
