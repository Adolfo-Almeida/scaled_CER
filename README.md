# Scaled-CER

Code for the scaled collaborative embedding regression (Scaled-CER) model published in our article **"The complementarity of a diverse range of deep learning features extracted from video content for video recommendation"**, **ESWA 2022**. The paper is available on [ScienceDirect](https://doi.org/10.1016/j.eswa.2021.116335), [ArXiv](https://arxiv.org/abs/2011.10834).

## Requirements

- TensorFlow == 1.15.*

## Execution

python main.py

## Acknowledgements

The Recommender Systems course at Polimi - https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi

## Reference

If you find our research useful, please cite our paper:

```
@article{ALMEIDA2022116335,
title = {The complementarity of a diverse range of deep learning features extracted from video content for video recommendation},
journal = {Expert Systems with Applications},
volume = {192},
pages = {116335},
year = {2022},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2021.116335},
url = {https://www.sciencedirect.com/science/article/pii/S095741742101633X},
author = {Adolfo Almeida and Johan Pieter {de Villiers} and Allan {De Freitas} and Mergandran Velayudan},
keywords = {Video recommendation, Deep learning features, Item cold-start, Item warm-start, Multimodal feature fusion, Beyond-accuracy metrics},
abstract = {Following the popularisation of media streaming, a number of video streaming services are continuously buying new video content to mine the potential profit from them. As such, the newly added content has to be handled well to be recommended to suitable users. In this paper, we address the new item cold-start problem by exploring the potential of various deep learning features to provide video recommendations. The deep learning features investigated include features that capture the visual-appearance, audio and motion information from video content. We also explore different fusion methods to evaluate how well these feature modalities can be combined to fully exploit the complementary information captured by them. Experiments on a real-world video dataset for movie recommendations show that deep learning features outperform hand-crafted features. In particular, recommendations generated with deep learning audio features and action-centric deep learning features are superior to MFCC and state-of-the-art iDT features. In addition, the combination of various deep learning features with hand-crafted features and textual metadata yields significant improvement in recommendations compared to combining only the former.}
}

```
