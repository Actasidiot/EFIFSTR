## Exploring Font-independent Features for Scene Text Recognition

This is the official Tensorflow implementation of the paper:

**Yizhi Wang and Zhouhui Lian. Exploring Font-independent Features for Scene Text Recognition. ACM Multimedia. 2020.**

The preprint and the [code](https://github.com/Actasidiot/EFIFSTR) will be released soon.

![teaser](img/teaser.PNG)

### Abstract

Scene text recognition (STR) has been extensively studied in last few years. Many recently-proposed methods are specially designed to accommodate the arbitrary shape, layout and orientation of scene texts, but ignoring that various font (or writing) styles also pose severe challenges to STR. These methods, where font features and content features of characters are tangled, perform poorly in text recognition on scene images with texts in novel font styles. To address this problem, we explore font-independent features of scene texts via attentional generation of glyphs in a large number of font styles. Specifically, we introduce trainable font embeddings to shape the font styles of generated glyphs, with the image feature of scene text only representing its essential patterns. The generation process is directed by the spatial attention mechanism, which effectively copes with irregular texts and generates higher-quality glyphs than existing image-to-image translation methods. Experiments conducted on several STR benchmarks demonstrate the superiority of our method compared to the state of the art.


## Model Architecture
![architecture](img/pipeline.PNG)

## Novel Font Scene Text (NFST) Dataset

As scene texts in novel font styles only make up a small proportion in existing benchmarks, we collect 100 text images with novel or unusual font styles to form a new dataset named as the Novel Font Scene Text (NFST) dataset.
<div align=center>
	<img src="img/NFSTdataset.jpg" width="500"> 
</div>

We compare our method with other two state-of-the-art methods ([ASTER](https://github.com/bgshih/aster) and [SAR](https://github.com/wangpengnorman/SAR-Strong-Baseline-for-Text-Recognition)) whose codes are publicly available. Our method significantly outperforms others on this dataset (see the following table), whose robustness to font style variance is proved.

| Training data        | Ours    |  SAR    |  ASETR  |
| -----                | ----    | ----    |----     |
| 90K+ST               | 55      |   45    | 44      |
| 90K+ST+SA+R          | 71      |   63    |  58     |

    <table>
        <tr>
            <th>设备</th>
            <th>设备文件名</th>
            <th>文件描述符</th>
            <th>类型</th>
        </tr>
        <tr>
            <th>键盘</th>
            <th>/dev/stdin</th>
            <th>0</th>
            <th>标准输入</th>
        </tr>
        <tr>
            <th>显示器</th>
            <th>/dev/stdout</th>
            <th>1</th>
            <th>标准输出</th>
        </tr>
        <tr>
            <th>显示器</th>
            <th>/dev/stderr</th>
            <th>2</th>
            <th>标准错误输出</th>
        </tr>
    </table>

