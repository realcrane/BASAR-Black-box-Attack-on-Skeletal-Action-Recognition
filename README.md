# BASAR-Black-box-Attack-on-Skeletal-Action-Recognition
BASAR:Black-box Attack on Skeletal Action Recognition, CVPR 2021

## Description

This is the source code of our CVPR 2021 paper: BASAR:Black-box Attack on Skeletal Action Recognition. BASAR is the first black-box adversarial attack approach for skeletal motions, which explores the interplay between the classifiation boundary and the natural motion manifold.  

Paper here: https://arxiv.org/abs/2103.05266

### Dependencies

Below is the key environment under which the code was developed, not necessarily the minimal requirements:

1. Python 3.7
2. pytorch 1.8.1
3. Cuda 11.2

And other libraries such as numpy and GEKKO. GEKKO is designed for large-scale optimization and accesses solvers of constrained, unconstrained, continuous, and discrete problems. The method to obtain GEKKO and tutorials can be found in https://apmonitor.com/wiki/index.php/Main/GekkoPythonOptimization. 

### Installing
No installation needed other than dependencies.

### Warning
The code has not been exhaustively tested. You need to run it at your own risk. The author will try to actively maintain it and fix reported bugs but this can be delayed.

### HDM05 demo

* The code assumes that you have normalised your data and know how to recover it after learning.

You can download the pre-processed data from [GoogleDrive](https://drive.google.com/file/d/1LyD-jf3X20wBbhKu071AwDDYvzFN7wjT/view?usp=sharing) or [BaiduYun(password:fmhm)](https://pan.baidu.com/s/1Itb94YjwUVqZmM9HLW6U3g) and extract files with
``` 
cd data
unzip <path to hdm05.zip>
```
