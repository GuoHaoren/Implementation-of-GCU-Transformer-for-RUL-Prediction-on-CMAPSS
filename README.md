# Implementation-of-GCU-Transformer-for-RUL-Prediction-on-CMAPSS
An implementation with GCU-Transformer with PyTorch for remaining useful life prediction on C-MAPSS.   
_Author: Haoren Guo, National University of Singapore_

This work is modified based on https://github.com/jiaxiang-cheng/PyTorch-Transformer-for-RUL-Prediction
## Quick Run
Simply modify the parameters in `train.sh` run `bash train.sh`. 

![image](https://user-images.githubusercontent.com/42372352/233019047-8a269673-f435-463c-a462-597b17c161a7.png)

## Testing
Change MODES='Train' to MODES='test' and change the MODEL_PATH to the model you saved. 

## Environment Details
```
python==3.8.8
numpy==1.20.1
pandas==1.2.4
matplotlib==3.3.4
pytorch==1.8.1
```

## Credit
This work is inpired by Mo, Y., Wu, Q., Li, X., & Huang, B. (2021). Remaining useful life estimation via transformer encoder enhanced by a gated convolutional unit. Journal of Intelligent Manufacturing, 1-10.

### If you find this repository beneficial to your research, I would appreciate it if you could cite relevant portions of my work.
```
@inproceedings{guo2022masked,
  title={Masked self-supervision for remaining useful lifetime prediction in machine tools},
  author={Guo, Haoren and Zhu, Haiyue and Wang, Jiahui and Vadakkepat, Prahlad and Ho, Weng Khuen and Lee, Tong Heng},
  booktitle={2022 IEEE 20th International Conference on Industrial Informatics (INDIN)},
  pages={353--358},
  year={2022},
  organization={IEEE}
}
```
```
@inproceedings{guo2023lightweight,
  title={Lightweight Compressed Temporal and Compressed Spatial Attention with Augmentation Fusion in Remaining Useful Life Prediction},
  author={Guo, Haoren and Zhu, Haiyue and Wang, Jiahui and Prahlad, Vadakkepat and Ho, Weng Khuen and de Silva, Clarence W and Lee, Tong Heng},
  booktitle={IECON 2023-49th Annual Conference of the IEEE Industrial Electronics Society},
  pages={1--6},
  year={2023},
  organization={IEEE}
}
```
```
@article{guo2024remaining,
  title={Remaining Useful Life Prediction via Frequency Emphasizing Mix-Up and Masked Reconstruction},
  author={Guo, Haoren and Zhu, Haiyue and Wang, Jiahui and Prahlad, Vadakkepat and Ho, Weng Khuen and de Silva, Clarence W and Lee, Tong Heng},
  journal={IEEE Transactions on Artificial Intelligence},
  year={2024},
  publisher={IEEE}
}
```
