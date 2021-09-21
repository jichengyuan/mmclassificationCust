<div align="center">
  <img src="resources/mmcls-logo.png" width="600"/> <img src="resources/Customization.jpg" width="250"/>
</div>

[![Build Status](https://github.com/open-mmlab/mmclassification/workflows/build/badge.svg)](https://github.com/open-mmlab/mmclassification/actions)
[![Documentation Status](https://readthedocs.org/projects/mmclassification/badge/?version=latest)](https://mmclassification.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/open-mmlab/mmclassification/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmclassification)
[![license](https://img.shields.io/github/license/open-mmlab/mmclassification.svg)](https://github.com/open-mmlab/mmclassification/blob/master/LICENSE)

## I. Quick-start on VNC ([based on example-annos](https://gitlab.tubit.tu-berlin.de/bifold/dataloader/tree/master/coco_anno_examples5))
#### 1) Open the existed Anaconda-env:
```shell
source activate open-mmlab
```
#### 2.1). Train from the pre-saved images and annotations in the path ./mmdetection/data/CocoCust:
    ├── mmdetection
        |── data
            ├── CocoCust
                └── annotations
                └── coco
                └── fakeKitti
```shell
python tools/train.py configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco_cust.py --gpus 1 --work-dir ./fcos_r50_fpn_gn_coco_cust
```

#### 2.2). Train from the pre-saved annotations and load images from other paths:
    ├── mmdetection
        |── data
            ├── CocoCust
                └── annotations
```shell
python tools/train.py configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco_cust_loadFromSeparatedFile.py --gpus 1 --work-dir ./fcos_r50_fpn_gn_coco_cust
```
Here, the example-data already exists in this repo.  

***Note 1.***: As an example, train2017 is same to val2017 which will indicate how the model overfit our training-data.

***&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2.***: Here, 'workers_per_gpu' was set to 0. If more workers were set, if will occurs an out of shared memory ERROR due to a docker environment configuretion. 

#### 3) Visualization in Tensorboard:
```shell
tensorboard --logdir=./fcos_r50_fpn_gn_coco_cust/tf_logs/ --bind_all
```

#### 4) Visualize the training progress:
```shell
http://127.0.0.1:6006/
```

## II. Quick-start on Pinky ([based on example-annos](https://gitlab.tubit.tu-berlin.de/bifold/dataloader/tree/master/coco_anno_examples5))
#### 1) Attch to the existed docker container:
```shell
docker attach mmdetection
```
#### 2) Other steps are same as on VNC:
 
***Note***: As an example, train2017 is same to val2017 which will indicate how the model overfit our training-data.
                       

## III.Below are mmdetection-toolbox Installation from [official repo](https://github.com/open-mmlab/mmdetection).
### Installation

Please refer to [get_started.md](docs/get_started.md) for installation.

### Getting Started

Please see [get_started.md](docs/get_started.md) for the basic usage of MMDetection.
We provide [colab tutorial](demo/MMDet_Tutorial.ipynb), and full guidance for quick run [with existing dataset](docs/1_exist_data_model.md) and [with new dataset](docs/2_new_data_model.md) for beginners.
There are also tutorials for [finetuning models](docs/tutorials/finetune.md), [adding new dataset](docs/tutorials/customize_dataset.md), [designing data pipeline](docs/tutorials/data_pipeline.md), [customizing models](docs/tutorials/customize_models.md), [customizing runtime settings](docs/tutorials/customize_runtime.md) and [useful tools](docs/useful_tools.md).

Please refer to [FAQ](docs/faq.md) for frequently asked questions.


## Acknowledgement

MMClassification is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new classifiers.