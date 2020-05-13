# Multiclass Semantic Segmentation HRNetV2 in Keras / Tensorflow

This repository serves as an educational example of implementing HRNetV2 in Keras / Tensorflow, including all supporting code one would need to apply it to arbitrary datasets. Datasets for training and validation are specified with text files pointing to original images, as well as color map segmentation masks. Color maps for each class are specified by a text file with "R G B" values on each line. Simple augmentation is supported for both training and inference steps. Loading pre-trained models for finetuning is included. Inference code is included that will generate output color maps that can be easily visualized.  

### Changelog:
- Added code in `segment_mc_semi.py` to include a consistency loss on a supplied unlabeled dataset for semi-supervised learning. 
- Added code in `segment_mc.py` to apply HRNet to arbitrary datasets specified by simple text files for training/validation and color maps.
- Removed other usage examples. Please refer to original repository for prior interface. 

