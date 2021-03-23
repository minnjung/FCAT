# FCAT: Fully Convolutional Network with Self-Attention for Point Cloud based Place Recognition

![Overview](media/overview.jpg)

### Abstract
Point cloud-based large-scale place recognition is still challenging due to the difficulty of extracting discriminative local descriptors from an unordered point cloud and integrating them effectively into a robust global descriptor. In this work, we construct a novel network named **FCAT (Fully Convolutional network with a self-ATtention unit)** that can generate a discriminative and context-aware global descriptor for place recognition from the 3D point cloud. It features with a novel sparse fully convolutional network architecture with sparse tensors for extracting informative local geometric features computed in a single pass. It also involves a self-attention module for 3D point cloud to encode local context information between local descriptors. Thanks to the effectiveness of these two modules, we demonstrate our method mostly outperforms state-of-the-art methods on large-scale place recognition tasks in PointNetVLAD. Moreover, our method shows strong robustness to different weather and light conditions through the experiments on the 6-DoF image-based visual localization task in RobotCar Seasons dataset.

### Environment and Dependencies
Code was tested using Python 3.8 with PyTorch 1.7 and MinkowskiEngine 0.4.3 on Ubuntu 18.04 with CUDA 10.2.

The following Python packages are required:
* PyTorch (version 1.7)
* MinkowskiEngine (version 0.4.3)
* pytorch_metric_learning (version 0.9.94 or above)
* tensorboard
* pandas
* psutil
* bitarray


Modify the `PYTHONPATH` environment variable to include absolute path to the project root folder: 
```export PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/home/.../FCAT
```

### Datasets

**FCAT** is trained on a subset of Oxford RobotCar and In-house (U.S., R.A., B.D.) datasets introduced in
*PointNetVLAD: Deep Point Cloud Based Retrieval for Large-Scale Place Recognition* paper ([link](https://arxiv.org/pdf/1804.03492)).
There are two training datasets:
- Baseline Dataset - consists of a training subset of Oxford RobotCar
- Refined Dataset - consists of training subset of Oxford RobotCar and training subset of In-house

For dataset description see PointNetVLAD paper or github repository ([link](https://github.com/mikacuy/pointnetvlad)).

You can download training and evaluation datasets from 
[here](https://drive.google.com/open?id=1rflmyfZ1v9cGGH0RL4qXRrKhg-8A-U9q) 
([alternative link](https://drive.google.com/file/d/1-1HA9Etw2PpZ8zHd3cjrfiZa8xzbp41J/view?usp=sharing)). 
Extract the folder in the same directory as the project code. Thus, in that directory you must have two folders: 1) benchmark_datasets and 2) FCAT

Before the network training or evaluation, run the below code to generate pickles with positive and negative point clouds for each anchor point cloud. 
 
```generate pickles
cd generating_queries/ 

# Generate training tuples for the Baseline Dataset
python generate_training_tuples_baseline.py

# Generate training tuples for the Refined Dataset
python generate_training_tuples_refine.py

# Generate evaluation tuples
python generate_test_sets.py
```

### Training
To train **MinkLoc3D** network, download and decompress the dataset and generate training pickles as described above.
Edit the configuration file (`config_baseline.txt` or `config_refined.txt`). 
Set `dataset_folder` parameter to the dataset root folder.
Modify `batch_size_limit` parameter depending on available GPU memory. 
Default limit (=256) requires at least 11GB of GPU RAM.

To train the network, run:

```train baseline
cd training

# To train FCAT model on the Baseline Dataset
python train.py --config ../config/config_baseline.txt --model_config ../models/fcat.txt

# To train FCAT model on the Refined Dataset
python train.py --config ../config/config_refined.txt --model_config ../models/fcat.txt
```

### Pre-trained Models

Pretrained models are available in `weights` directory
- `fcat_baseline.pth` trained on the Baseline Dataset 
- `fcat_refined.pth` trained on the Refined Dataset 

### Evaluation

To evaluate pretrained models run the following commands:

```eval baseline
cd eval

# To evaluate the model trained on the Baseline Dataset
python evaluate.py --config ../config/config_baseline.txt --model_config ../models/fcat.txt --weights ../weights/fcat_baseline.pth

# To evaluate the model trained on the Refined Dataset
python evaluate.py --config ../config/config_refined.txt --model_config ../models/fcat.txt --weights ../weights/fcat_refined.pth
```

## Results

**FCAT** performance (measured by Average Precision@1\%) compared to state-of-the-art:

### Trained on Baseline Dataset

| Method         | Oxford  | U.S. | R.A. | B.D |
| ------------------ |---------------- | -------------- |---|---|
| PointNetVLAD [1] |     80.3     |   72.6 | 60.3 | 65.3 |
| PCAN [2] |     83.8     |   79.1 | 71.2 | 66.8 |
| DH3D-4096 [3] | 84.3 | - | - | - |
| DAGC [4] |     87.5     |   83.5 | 75.7 | 71.2 |
| LPD-Net [5] |     94.9   |   96.0 | 90.5 | 89.1 |
| MinkLoc3D [6]  |     97.9     |   95.0 | 91.2 | 88.5 |
| FCAT (ours) | **98.2** | **96.4** | **94.0** | **91.7** |


### Trained on Refined Dataset

| Method         | Oxford  | U.S. | R.A. | B.D |
| ------------------ |---------------- | -------------- |---|---|
| PointNetVLAD [1] |     80.1     |   94.5 | 93.1 | 86.5 |
| PCAN [2] |     86.4     |   94.1 | 92.3 | 87.0 |
| DAGC [4] |     87.8     |   94.3 | 93.4 | 88.5 |
| LPD-Net [5] |     94.9     |   98.9 | 96.4 | 94.4 |
| MinkLoc3D [6]  |     **98.5**     |   99.7 | **99.3** | **96.7** |
| FCAT (ours) | 98.3 | **99.8** | 98.7 | **96.7**|

1. M. A. Uy and G. H. Lee, "PointNetVLAD: Deep Point Cloud Based Retrieval for Large-Scale Place Recognition," 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
2. W. Zhang and C. Xiao, "PCAN: 3D Attention Map Learning Using Contextual Information for Point Cloud Based Retrieval," 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
3. Du, Juan, Rui Wang, and Daniel Cremers. "DH3D: Deep Hierarchical 3D Descriptors for Robust Large-Scale 6DoF Relocalization." European Conference on Computer Vision (ECCV). Springer, Cham, 2020.
4. Q. Sun et al., "DAGC: Employing Dual Attention and Graph Convolution for Point Cloud based Place Recognition", Proceedings of the 2020 International Conference on Multimedia Retrieval
5. Z. Liu et al., "LPD-Net: 3D Point Cloud Learning for Large-Scale Place Recognition and Environment Analysis," 2019 IEEE/CVF International Conference on Computer Vision (ICCV)
6. J. Komorowski, "MinkLoc3D: Point Cloud Based Large-Scale Place Recognition", Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), (2021)
