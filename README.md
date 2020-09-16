## Exploring Font-independent Features for Scene Text Recognition

This is the official Tensorflow implementation of the paper:

**Yizhi Wang and Zhouhui Lian. Exploring Font-independent Features for Scene Text Recognition. ACM Multimedia. 2020.**

![teaser](img/teaser.PNG)

## Novel Font Scene Text (NFST) Dataset

As scene texts in novel font styles only make up a small proportion in existing benchmarks, we collect 100 text images with novel or unusual font styles to form a new dataset named as the Novel Font Scene Text (NFST) dataset ([download link](https://raw.githubusercontent.com/Actasidiot/EFIFSTR/master/NFST.zip)).
<div align=center>
	<img src="img/NFSTdataset.jpg" width="500"> 
</div>

We compare our method with other two state-of-the-art methods ([ASTER](https://github.com/bgshih/aster) and [SAR](https://github.com/wangpengnorman/SAR-Strong-Baseline-for-Text-Recognition)) whose codes are publicly available. Our method significantly outperforms others on this dataset (see the following table), whose robustness to font style variance is proved.

<div align=center>
<table>
    <thead>
        <tr>
            <th>Training data</th>
            <th>Ours</th>
            <th>SAR</th>
	    <th>ASETR</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th>90K+ST</th>
            <th>55</th>
            <th>45</th>
	    <th>44</th>
        </tr>
        <tr>
            <th>90K+ST+SA+R</th>
            <th>71</th>
            <th>63</th>
	    <th>58</th>
        </tr>
    </tbody>
</table>
</div>

## Prerequisites

 **TensorFlow r1.15**.



## Installation
  1. Go to `c_ops/` and run `build.sh` to build the custom operators
  2. Execute `protoc protos/*.proto --python_out=.` to build the protobuf files
