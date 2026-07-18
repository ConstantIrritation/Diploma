
# Я это обязательно заполню, пока тут будет ридми для статьи
<!-- # Convolution-Centering-Semantic-Analysis -->

## 📄 Abstract

Despite the widespread using of Normalization Layers for stabilizing the deep networks training, the semantic role of the centering operation remains insufficiently explored. Modern research is largely focused on the optimization aspects e.g. of the Batch Normalization Layer. We postulate that centering actively suppresses activation components that are linearly dependent on the batch mean, which mostly correspond to domain-specific rather than class-specific features. To test this hypothesis, we analyze the interaction between the mean vector and the layer weights, and track the dynamics of cosine similarities within and between classes across a wide range of tasks and architectures. Specifically, we examine: a YOLO model on the COCO dataset, the MatchboxNet model for Keyword Spotting, and ResNet models for face recognition and image classification. The results empirically demonstrate that depth-wise centering hierarchically filters common patterns, enhances class-specific features, and improves class compactness in the activation space. Our findings reveal the nature of suppressing semantically common components allows us to use the center vector for semantic coloring of weights and opens new research directions for the interpretation analysis of representations in deep networks.

## Repository Structure
 
```
Convolution-Centering-Semantic-Analysis/
│
├── plots_genaral.ipynb                # Figure 6: cross-model summary plots
│
├── yolo/                              # DAMO-YOLO experiments (COCO dataset)
│   ├── plots_yolo.ipynb              # Figures 1, 7, 9, 10; Figures 6, 8 preparation
│   ├── configs/
│   │   └── damoyolo_tinynasL20_T.py  # Model & dataset config
│   ├── tools/
│   │   └── calculate_angles_yolo.py  # computes per-layer cosine similarities
│   ├── my_help_functions/
│   │   ├── hooks.py                  # Forward-hook registration
│   │   ├── datasets.py               # CustomCocoDataset wrapper
│   │   └── visualise_arch.py         # Architecture graph utilities
│   ├── damo/                         # DAMO-YOLO utils from original repository
│   └── requirements.txt
│
├── facerec_and_cifar/                 # ResNet experiments (face recognition + CIFAR-100)
│   ├── plots_arcface.ipynb           # Figures 6, 8 preparation
│   ├── plots_cifar100.ipynb          # Figures 6, 8 preparation
│   ├── tools/
│   │   ├── calculate_angles_facerec.py
│   │   └── calculate_angles_cifar.py
│   ├── my_help_functions/
│   │   ├── hooks.py
│   │   └── utils.py                  # get_sampled_cos_paired, ScalableMask
│   ├── pytorch_cifar100/             # CIFAR-100 training utilities from original repository
│   └── facerec_requirements.txt
│
└── sound/                             # MatchboxNet experiments (Keyword Spotting)
    ├── plots_sound.ipynb             # Figures 6, 8 preparation
    ├── train.ipynb                   # Training example with different shifting (i.e. no mean and no bias)
    ├── my_help_functions/
    │   ├── hooks.py                  
    │   ├── utils.py                  
    │   └── calculate_angles_sound.py
    └── install.sh
```

## Reproducing the results


### Pipeline description

<details>
<summary></summary>

This description refers to the YOLO experiment, while others follow the same general procedure.

### Hook registration

`get_and_save_res()` iterates over consecutive `(Conv2d, BatchNorm2d)` pairs and registers a forward hook via `register_hooks()`.

The hook extracts activation snapshots. For example, `after conv` or `before and after centering`.

### Fig. 8 experiments

For Fig. 8, the hook performs the following steps:

1. **SVD of the weight matrix**
2. **Rank selection via correlation type**: rank singular directions either by BN mean projected into the singular basis `Uᵀ · μ` or by singular value magnitude `S`, then keep a subset according to a threshold.
3. **Zeroing and rescaling**: remove selected singular directions and rescale the remaining ones.
4. **Re-projection**: apply a new convolution with the modified weight and return the original and modified activations.

### Calculate cosine similarities

After the forward pass, for each pair of classes, including the background, the cosine similarity is calculated using the `get_sampled_cos()` function.

Results are accumulated across batches and saved in `heap/angles/<experiment_name>.pkl`.

</details>

### Installation

<details>
<summary>DAMO-YOLO</summary>

Step1. Install DAMO-YOLO.
```shell
git clone https://github.com/vshokorov/Convolution-Centering-Semantic-Analysis.git
cd Convolution-Centering-Semantic-Analysis/YOLO
conda create -n DAMO-YOLO python=3.7 -y
conda activate DAMO-YOLO
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PYTHONPATH
```
Step2. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip install cython;
pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI # for Linux
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI # for Windows
```

Step3. Download a pretrained torch from [the benchmark table](https://github.com/tinyvision/damo-yolo#model-zoo) for Tiny model: damoyolo_tinynasL20_T_420.pth
</details>

<details>

<summary>Audio</summary>

Step1. Install environment.
```shell
git clone https://github.com/vshokorov/Convolution-Centering-Semantic-Analysis.git
cd Convolution-Centering-Semantic-Analysis/sound
chmod +x install.sh
./install.sh
# This will create new "sound" conda environment with all dependencies
```
Step2. Prepare dataset.

```shell
# requires 7 GB of free space
unzip sound/sound_dataset_demo/google_speech_recognition_v1.zip -d sound/sound_dataset_demo/google_speech_recognition_v1
unzip sound/sound_dataset_demo/google_speech_recognition_v2_1.zip -d sound/sound_dataset_demo/google_speech_recognition_v2
unzip sound/sound_dataset_demo/google_speech_recognition_v2_2.zip -d sound/sound_dataset_demo/google_speech_recognition_v2
```

</details>

<details>

<summary>FaceRec</summary>

Step1. Install environment.

**Follow the installation instruction for YOLO, then**
```shell
cd ../facerec_and_cifar
pip install -r facerec_requirements.txt
```
Step2. Prepare dataset.

```shell
unzip datasets
```

</details>

<details>

<summary>Cifar100</summary>

Step1. Install environment.

**Follow the installation instruction for YOLO, then**
```shell
cd ../facerec_and_cifar
```

</details>


### Running Experiments
To reproduce any results check **plots_general.ipynb** for further instuctions
