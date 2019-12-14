# Depth Estimation in Nighttime using Stereo-Consistent Cyclic Translations, arXiv'19
PyTorch code for the paper - Depth Estimation in Nighttime using Stereo-Consistent Cyclic Translations, arXiv'19

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

### Training
To be released soon. 

### Testing
To generate sample results on the Oxford RobotCar dataset, run for e.g. 
```
$ python predict.py --imgname 00701 --datapath ./sampledata/ --ckptpath ./pretrained_ckpts/
```
Pre-trained checkpoints for the generators and stereo networks are provided for generating the disparity results. 

### Sample Results
Image: 00094 (Poorly-Lit)
![sample_0001_18](images/0001_18_f.png)

Image: 00701 (Well-Lit)
![sample_0006_06](images/0006_06_f.png)
  
### Acknowledgements 
We are thankful to the authors of [PSMNet](https://github.com/JiaRenChang/PSMNet) and [ToDayGAN](https://github.com/AAnoosheh/ToDayGAN) for making their code public. 
