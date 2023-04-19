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

