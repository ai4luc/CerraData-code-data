# A High-Spatial Resolution Dataset and Few-shot Deep Learning Benchmark for Image Classification: Source Code

Source code for the **dataset generation** and **experimental evaluation** of our paper *[A High-Spatial Resolution Dataset and Few-shot Deep Learning Benchmark for Image Classification](10.1109/SIBGRAPI55357.2022.9991746})*.

## Biome Cerrado Dataset (CerraData)

The Biome Cerrado Dataset (CerraData) was generated using the public available satellite data from [INPE](http://www.dgi.inpe.br/). We firstly obtained 150 scenes made by the *Wide Panchromatic and Multispectral Camera (WPM)* of the CBERS-4A satellite. Each scene was preprocessed with the [merge bands algorithm](data_management/merge_bands.py) in order to merge the near-infrared (NIR), green (G) and blue (B) spectral bands, respectively, to the R, G and B image channels. In sequence, by the use of [QGIS platform](https://qgis.org/pt_BR/site/), pan-sharpening with Hue Saturation Value (HSV) method via panchromatic (PAN) band was also applied to the merged scenes. Then, [cut and filter algorithm](data_management/cut_and_filter_images.py) was applied to the scenes, which crops tiles of 256x256 pixels that preserves geospatial information, and also remove non-data images. At this point, approximatelly 2.5M usable tiles was produced.

![image](classes.jpeg)

The final step was, based on [Neves et. al.](https://doi.org/10.1117/1.JRS.15.044504) and [Ribeiro et. al.](https://www.embrapa.br/busca-de-publicacoes/-/publicacao/554094/fitofisionomias-do-bioma-cerrado), manually label 50k images from the hole 2.5M population. For this purpose 50k images  were randomly choosen (in a balanced manner, i.e., 10k images per class) and labeled in 5 different classes: 
1. **Cultivated Area:** samples comprising pasture, agriculture, and planted trees; 
2. **Forest Formation:** samples are characterized by the predominance of arboreal formation and riparian forests; 
3. **Non-Forest Area:** images of urban areas, mining, fire scars, and dunes; 
4. **Savanna Formation:** samples of five different phytophysiognomies, i.e., woodland savanna, typical savanna, rupestrian savanna, shrub savanna, and vereda; and 
5. **Water:** river, small lakes, dams, and fish farming ponds.

### Download the dataset
Currently, the labeled part of the dataset (50k images) is available to download via this [cloud storage link](https://www.kaggle.com/datasets/cerranet/biome-cerrado-dataset-cerradata). The full unlabeled dataset is available to download in this [cloud drive](https://inpebr-my.sharepoint.com/:f:/g/personal/mateus_miranda_inpe_br/EhAvFUXWZVlGq_saQc_wPXcB-5x5wwM_9wi4dkhzGMD9pA?e=K1H5bt), which is organized in five folders corresponding to Brazilian states. For each folder four images batch was created considering its geoinformation. 

## Dataset challenge assessment: Evaluation of well-known CNNs

This repo contains the source code used to obtain all the results presented on the paper. Following will be shown (1) how to simply use the code and also (2) how to reproduce our presented results.

### How to use the code

To simply run the source code, do the following steps:

1. Install the following requirements:
    
    ```
    python==3.8 
    CUDA==11.6.1
    torch==1.12
    torchvision==0.13
    scikit-learn
    tensorboard  #optional
    umap-learn
    tqdm
    ```


2. **Clone** the repo and **download** the dataset. By default the running code will search for the dataset in `data/cerradov3` directory inside the repo.

    >  It can be changed by your needs, only adjust the `--root` option of the `trainer.py` code. In this case we highly recommend to put the dataset main directory (`cerradov3`) inside some parent directory (eg. in our case `./data`) because a lot of auxiliary files about the dataset will be generated and saved along the dataset.

3. Run the following command to train the DL model (ResNet-18 by default) and evaluate the last trained model on the test split.

    `python trainer.py`

    > `trainer.py` has a lot of different command line options used to adjust the training hyperparameters, dataset splitting, and some other miscelaneous adjustments. We highlight the `--arch` option, used to select in between different available CNNs architectures, and `--runs` option, to run the experiment $N$ times and compute the average and standard error of the performance measurement. Please, use `-h` to see other available configurations.

4. Evaluate the best trained model checkpoint by running the following command. Ensure to do not change any command line parameter used during the training step since the code will automatically search for the best model in the right path.

    `python trainer.py --evaluate --resume`

### Reproducing our results

We tried to be quite strict about reproducibility. If you are aiming to rerun our experiments in the same conditions as us, please, download the PyTorch Docker Container (tag version #22.04-py3) from [NVIDIA NGC catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch),and with some container software of you choice (eg. Docker and **Singularity**) run the code inside the container.

Moreover, make sure to download our dataset splits from this [cloud storage link](https://drive.google.com/drive/folders/1R9QNvLu60WKtsgRzjaXz3pPnmRrcACkb?usp=sharing) and place them (all `.npz` files) in the parent directory of the dataset `cerradov3` (in our case the `data/` directory inside the repo). 

We made available all submission [scripts](scripts/) used to run our experiments. Note that they were made based on the Singularity container software and must be ran in a target computational system managed by the **SLURM resource management software**. If you meet this requirements, please, use the following command:
    
`sbatch scripts/run_sbatch_<model>.slurm`

> Do not forget to adjust the **queue** name and update the path of the submission scripts based in your own environment, such as, the absolute path to the (1) repo `/path/for/the/repo/CerraData-code-data` in the `-B` option, and (2) the path to the container image file `/path/for/the/singularity/file/container.sif`.

## Acknowledgments

This research was developed within the [**IDeepS**](https://github.com/vsantjr/IDeepS) project which is supported by the Brazilian LNCC/MCTI via resources of the [SDumont supercomputer](http://sdumont.lncc.br). This research was also supported by the Brazilian agencies CAPES (grants #88887.603929/2021-00, #88887.470055/2019-00), FAPESP (grant #2020/08770-3), and CNPq (grants #303360/2019-4, #314868/2020-8).

## Reference

MIRANDA, M. S.; SILVA, L. F. A.; SANTOS, S. F.; SANTIAGO JÚNIOR, V. A.; KÖRTING, T. S.; ALMEIDA, J. A High-Spatial Resolution Dataset and Few-shot Deep Learning Benchmark for Image Classification. In: CONFERENCE ON GRAPHICS, PATTERNS AND IMAGES, 35. (SIBGRAPI), 2022, Natal, RN. Proceedings... 2022. On-line. IBI: <8JMKD3MGPEW34M/47JU8TS>. Available from: <http://urlib.net/ibi/8JMKD3MGPEW34M/47JU8TS>. 

