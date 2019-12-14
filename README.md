# Depth Estimation in Nighttime using Stereo-Consistent Cyclic Translations, arXiv'19
PyTorch code for the paper - Depth Estimation in Nighttime using Stereo-Consistent Cyclic Translations, arXiv'19, Aashish Sharma, Robby T. Tan, and Loong-Fah Cheong. 

Please cite the paper if you find this code useful:
```
@article{sharma2019depth,
  title={Depth Estimation in Nighttime using Stereo-Consistent Cyclic Translations},
  author={Sharma, Aashish and Tan, Robby T and Cheong, Loong-Fah},
  journal={arXiv preprint arXiv:1909.13701},
  year={2019}
}
```
### Requirements
The code is tested on Python 3.7, PyTorch 1.1.0, TorchVision 0.3.0. 

### Prediction
To generate sample results on the Oxford RobotCar dataset, run for e.g. 
```
$ python predict.py --imgname 00701 --datapath ./sampledata/ --ckptpath ./pretrained_ckpts/
```
Pre-trained checkpoints for the generators and stereo networks are provided for generating the disparity results. 

### Sample Results
The results shown below are in the following order: Left Image, Right Image, Disaprity Result from PSMNet, and Disparity Result Proposed. 

Image: 00094 (Poorly-Lit Image)
![00094](images/00094_f.png)

Image: 00701 (Well-Lit Image)
![00701](images/00701_f.png)
  
### Acknowledgements 
Thanks to the authors of [PSMNet](https://github.com/JiaRenChang/PSMNet) and [ToDayGAN](https://github.com/AAnoosheh/ToDayGAN) for making their code public. 
