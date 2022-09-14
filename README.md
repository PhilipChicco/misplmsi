# Microsatellite Instability (MSI) detection/classification in Whole Slide Images

<div align="center">
<img width="100%" alt="pipeline-full" src="1.PNG">
</div>

## Enviroment Requirements

* Ubuntu 20
* Python 3.6/3.7
* [CUDA 11.0](https://developer.nvidia.com/cuda-toolkit)
* [Pytorch 1.7.1](https://pytorch.org)
* [Openslide](https://github.com/openslide/openslide-python)

## Conda environment installation

````bash
conda env create --name msihisto python=3.6
conda activate msihisto
```
````

* run `pip install -r requirements.txt`

## Code Structure

The code base is built to be modular and allow for easy extension. All scripts should be relative to the top directory i.e.,

````bash
cd misplmsi/
python research_mil/xxx.py 

````

```bash
research_mil/
-- configs/ : defines the config files for pre-processing, training and inference
-- data/ : consists of data splits (text files) and created lib files 
-- evaluators/ : inference scripts for model testing and collection of the main results
-- trainers/ : base trainer interfaces and model specific training scripts
-- models /: consists of network definitions
-- result_utils & utils/ : Some useful methods 
-- loaders/ : defines the data loaders.
-- wsi_tools/ : collection of pre-processing scripts.
-- logs/ : default folder to save models and results. (can be changed to other location)
train.py: the main training script that requires a config file
test.py : the main testing script.
```

## WSI Data Preprocessing

* Refer to the [ReadMe](wsi_tools/README.md) file in research_mil/wsi_tools

## Train|Test MSI patch encoder

* To train the MSI or Tumor detector use the config file 'configs/amc_2020/patch_cls'.
* The config file defines all the necessary parameters and model options i.e. loads a resnet34 model with a single classification head (see. models/pooling/classic.py)
* Also, you need to modify the paths to the patches extracted in the pre-processing step.
* By default, the model is trained for 20 epochs with Adam optimizer (intial LR: 1e-5) decayed at epochs 5,10,15). Note that the backbone (feature model) and classifier (FC) have different learning rates.
* The config file defines a logging directory (logdir) to save the models during training and includes a logdir for testing (which model to load in inference).
* To train and test (patche level) run:

```python
python research_mil/train.py --config/amc_2020/patch_cls.yml
python research_mil/test.py --config/amc_2020/patch_cls.yml
```

* All results test results are saved in the testting-logdir specified in the config file. Results include AUC,Accuracy,TPR,FPR and so on. In addition, confusion matrix is included.
* To use pre-trained models, adjust the patches for testing checkpoints in the config file. (carefully check)
* All pre-trained patch encoder weights are stored in './research_mil/logs/pretrained/patch_encoder': To test on new data only run the test.py script.

## Train|Test MSI WSI model

The WSI model requires features from the trained (patch-encoder) i.e., the patch encoder will produce a feature map of 1x512x8x8 for a single patch input.

Consequently, the WSI Model aggregates the features of all patches in a bag (Kx512x8x8) to (1xD) using mean convolutional pooling, and has a single classification head for MSI/MSS prediction.

See models module for details on the architecture i.e., models/pooling/deepset.py. To train the WSI model run:

```python
python research_mil/train.py --config/amc_2020/deepset.yml
python research_mil/test.py --config/amc_2020/deepset.yml
```

See the config file /configs/amc_2020/deepset.yml for more details. By default, during training k=32 patches are sampled per WSI to create bags. To adjust the default setting, the train_batch_size should be set to 1; in order to load larger bags. However, during testing the default option is k=512.

In order to use the pre-trained models, uncomment the paths in the config file. All pre-trained models are saved in './research_mil/logs/pretrained/wsi_model'. Simply run test.py only for new data when using the pretrained models.

## WSI MSI Segmentation



<div class="row">
  <div class="column">
    <img src="2.png" alt="Snow" style="width:50%">
  </div>
  <div class="column">
    <img src="3.png" alt="Forest" style="width:50%">
  </div>
</div>

* For the task of patch-based WSI segmentation of tumor or MSI regions, the script wsi_tools/wsi_mil_seg.py should be used.
* **wsi_mil_seg.py** has two settings: (i) subtype segmentation of MSS/MSI using reference tumor mask (ground-truth tumor regions) and (ii) tumor segmentation on the entire tissue regions.
* In addition, each setting can produce either sparse or dense segmentation maps similar to the figures above. Figure: (Top) Dense segmentation of MSI, and (Below) Ground-truth tumor region (yellow).
* To run WSI MSI segmentation using ground-truth reference masks, first modify the config file (**configs/wsi_seg.yml**). The paths to the masks (e.g. WSI_ID_tumor.png produced in the pre-processing) location,  the patch-encoder (tumor/msi detector) path, and library file containing the slide paths should be specified.
* To perform sparse/dense segmentation;

````bash
# Perform dense subtype segmentation
# the flags --subtype and --full should be included 
python research_mil/wsi_tools/wsi_mil_seg.py --config /path/to/configs/amc_2020/wsi_seg.yml --subtype --full 

# To perform sparse segementation (do not include the option --full)
python research_mil/wsi_tools/wsi_mil_seg.py --config /path/to/configs/amc_2020/wsi_seg.yml --subtype

# The algorithm will save the heatmaps(.png), tumor reference mask overlay (.png) and probability map (.numpy)
# To perform standard tumor segmentation in sparse or full mode (use flag --full)
python research_mil/wsi_tools/wsi_mil_seg.py --config /path/to/configs/amc_2020/wsi_seg.yml
```
````

* The segmentation algorithm samples overlapping patches from the WSI image using (loaders/datasets.py: **GridWSIPatchDataset** class). All patches are imagenet normalized after stain correction. Please refer the files for more details.

## Extensions & Improvements

* For weakly supervised learning, ensure that you are only extracting patches from the tissue region.
* Any other improvements are encouraged.
* Good Luck!
