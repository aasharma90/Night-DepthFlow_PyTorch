# Depth Estimation in Nighttime using Stereo-Consistent Cyclic Translations, arXiv'19 (for Optical Flow)
PyTorch code for the nighttime flow application of the [paper](https://arxiv.org/abs/1909.13701) - Depth Estimation in Nighttime using Stereo-Consistent Cyclic Translations, arXiv'19, Aashish Sharma, Robby T. Tan, and Loong-Fah Cheong. 

For original nighttime depth from stereo code, please refer [here](https://github.com/aasharma90/NighttimeDepthandFlow/tree/master/CycleStereoGAN_NighttimeDepth)

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

During training, memory consumption per GPU (keeping a batch size of 4) was observed to be ~10Gb. 


### Prediction
For sample testing, run for e.g. 
```
$ python predict.py --imglist ./datafiles/sample_nighttime_data1_01.txt --resultpath ./results/nighttime_data1/
```
Pre-trained checkpoints for the generators and flow networks are provided for generating the flow results. The results directory will contain both the `.flo` and visualized flow results. 

### Sample Results
The results shown below are in the following order: Image-1, Flow Result from PWCNet, and Flow Result Proposed. 

  
### Acknowledgements 
Thanks to the authors of [PWCNet](https://github.com/NVlabs/PWC-Net/tree/master/PyTorch) and [ToDayGAN](https://github.com/AAnoosheh/ToDayGAN) for making their code public. 
