# StaSiS-Net
### StaSiS-Net: a stacked and siamese disparity estimation network for depth reconstruction in modern 3D laparoscopy.

<p align="center">
  <img width="1000" height="400" src="https://github.com/lodeguns/StaSiS-Net/blob/main/imgs/visual_abstract.png?raw=true">
</p>


### Abstract
Accurate and real-time methodologies for a non-invasive three-dimensional representation and reconstruction of internal patient structures is one of the main research fields in computer-assisted surgery and endoscopy.  Mono and stereo endoscopic images of soft tissues are converted into a three-dimensional representation by the estimation of depth maps.  However, automatic, detailed, accurate and robust depth map estimationis a challenging problem which, moreover, is strictly dependent on a robust estimate of  the  disparity  map.   Many  traditional  algorithms  are  often  inefficient  or  not  accurate.  In this work, novel self-supervised stacked and Siamese encoder/decoder neural networks are proposed to compute accurate disparity maps for 3D laparoscopy depth reconstructions.   These  networks  produce  disparities  in  real-time  on  standard  GPU-equipped desktop computers and after,  with a minimal parameter configuration their depth reconstruction.  We compare their performance on three different public datasets and on a new challenging simulated dataset and they outperform state-of-the-art mono and stereo depth estimation methods.  Extensive robustness and sensitivity analyses on more than 30 000 frames has been performed.  This work leads to important improvements in mono and stereo real-time depth estimations of soft tissues and organs with a very low average mean absolute disparity reconstruction error with respect to ground truth.

### Supplementary Materials
This repository contains a part of the Supplementary Material related to the paper: StaSiS-Net: a stacked and siamese stereo network 
for depth estimation in modern 3D laparoscopy - Bardozzo F., Collins T., Hostettler A., Forgione A. and Tagliaferri R. DOI https://doi.org/10.1016/j.media.2022.102380

For more information, refer to the paper and its PDF supplement. However, additional samples and examples from our simulated dataset of internal surfaces are linked [here](https://drive.google.com/drive/folders/12Q3qrlFGaBd6R2wcISjx-XgN9t5WwXfe).  Further, it is provided the 3D model with camera intrinsics used for simulation.  

**Model Prediction**
Already trained models can be tested by downloading [this repo](https://drive.google.com/drive/folders/1_atwJnYU61aGYjrKrhh8s32mgfpzYdhh?usp=sharing).

**Model Training**
The file to perform training with several configurations is provided.

**Simulated Dataset and Real Datasets**
The datasets are available upon request and only for research collaborations.


If you use this code in the training or prediction phase you must mention 
its origin by citing the reference papers.


For a visual quality assessment, on this [YouTube video](https://www.youtube.com/watch?v=TiX3eXXbcbQ) 
some 3D reconstructions on simulated and real videos are provided.


<p align="center">
  <img width="600" height="700" src="https://github.com/lodeguns/StaSiS-Net/blob/main/imgs/gh_example.png?raw=true">
</p>


**How to cite this paper**

```
@article{Bardozzo2022stasis,
title = {StaSiS-Net: a stacked and siamese disparity estimation network for depth reconstruction in modern 3D laparoscopy.},
journal = {Medical Image Analysis},
pages = {102380},
year = {2022},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2022.102380},
url = {https://www.sciencedirect.com/science/article/pii/S1361841522000329},
author = {Francesco Bardozzo and Toby Collins and Antonello Forgione and Alexandre Hostettler and Roberto Tagliaferri}
}
```

```
@inproceedings{bardozzo2022cross,
  title={Cross X-AI: Explainable Semantic Segmentation of Laparoscopic Images in Relation to Depth Estimation},
  author={Bardozzo, Francesco and Priscoli, Mattia Delli and Collins, Toby and Forgione, Antonello and Hostettler, Alexandre and Tagliaferri, Roberto},
  booktitle={2022 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--8},
  year={2022},
  organization={IEEE}
}
```

**Licence**
The same of the Medical Image Analysis - Journals | Elsevier

This work is supported by the Artificial Intelligence departement DISA-MIS, NeuRoNe Lab (University of Salerno - IT) and IRCAD (Research Institute against Digestive Cancer - Strasbourg - FR)
