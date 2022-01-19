# StaSiS-Net
### StaSiS-Net: a stacked and siamese stereo network for depth reconstruction in modern 3D laparoscopy.

### Abstract
Accurate and real-time methodologies for a non-invasive three-dimensional representa-tion and reconstruction of internal patient structures is one of the main research fieldsin computer-assisted surgery and endoscopy.  Mono and stereo endoscopic images ofsoft tissues are converted into a three-dimensional representation by the estimation ofdepth maps.  However, automatic, detailed, accurate and robust depth map estimationis a challenging problem which, moreover, is strictly dependent on a robust estimateof  the  disparity  map.   Many  traditional  algorithms  are  often  inefficient  or  not  accu-rate.  In this work, novel self-supervised stacked and Siamese encoder/decoder neuralnetworks are proposed to compute accurate disparity maps for 3D laparoscopy depthreconstructions.   These  networks  produce  disparities  in  real-time  on  standard  GPU-equipped desktop computers and after,  with a minimal parameter configuration theirdepth reconstruction.  We compare their performance on three different public datasetsand on a new challenging simulated dataset and they outperform state-of-the-art monoand stereo depth estimation methods.  Extensive robustness and sensitivity analyses onmore than 30 000 frames has been performed.  This work leads to important improve-ments in mono and stereo real-time depth estimations of soft tissues and organs with avery low average mean absolute disparity reconstruction error with respect to groundtruth.

### Supplementary Materials
This repository contains a part of the Supplementary Material related to the paper: StaSiS-Net: a stacked and siamese stereo network 
for depth estimation in modern 3D laparoscopy - Bardozzo F., Collins T., Hostettler A., Forgione A. and Tagliaferri R.

For more information, refer to the paper and its PDF supplement. However, additional samples and examples from our simulated dataset of internal surfaces are linked [here](https://drive.google.com/drive/folders/12Q3qrlFGaBd6R2wcISjx-XgN9t5WwXfe).  Further, it is provided the 3D model with camera intrinsics used for simulation.  

For a visual quality assessment, on this [YouTube video](https://www.youtube.com/watch?v=TiX3eXXbcbQ) 
some 3D reconstructions on simulated and real videos are provided.


<p align="center">
  <img width="600" height="700" src="https://github.com/lodeguns/StaSiS-Net/blob/main/imgs/gh_example.png?raw=true">
</p>




In the meantime we update the repository both the code and the dataset are available upon request 
(Dr. Francesco Bardozzzo - fbardozzo at unisa dot it).


**Licence**
The same of the Medical Image Analysis - Journals | Elsevier

This work is supported by the Artificial Intelligence departement DISA-MIS, NeuRoNe Lab (University of Salerno - IT) and IRCAD (Research Institute against Digestive Cancer)
