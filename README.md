<img src="image/USC-Logos.png" width=120px /><img src="./image/Adobe-Logos.png" width=120px />

<div align="center">

# Can3Tok: Canonical 3D Tokenization and Latent Modeling of Scene-Level 3D Gaussians

### ICCV 2025

<p align="center">  
    <a href="https://zerg-overmind.github.io/">Quankai Gao</a><sup>1</sup>,
    <a href="https://iliyan.com/">Iliyan Georgiev</a><sup>2</sup>,
    <a href="https://tuanfeng.github.io/">Tuanfeng Y. Wang</a><sup>2</sup>,
    <a href="https://krsingh.cs.ucdavis.edu/">Krishna Kumar Singh</a><sup>2</sup>,
    <a href="https://viterbi.usc.edu/directory/faculty/Neumann/Ulrich">Ulrich Neumann</a><sup>1+</sup>,
    <a href="https://gorokee.github.io/jsyoon/">Jae Shin Yoon</a><sup>2+</sup>
    <br>
    <sup>1</sup>USC <sup>2</sup>Adobe Research
</p>

</div>
In this project, we introduce Can3Tok, the first 3D scene-level variational autoencoder (VAE) capable of encoding a large number of Gaussian primitives into a low-dimensional latent embedding, which enables high-quality and efficient generative modeling of complex 3D scenes.

## Overall Instruction
1. We firstly run structure-from-motion (SfM) on [DL3DV-10K](https://github.com/DL3DV-10K/Dataset) dataset with [COLMAP](https://colmap.github.io/) to get the camera parameters and sparse point clouds i.e. SfM points. 
2. Then, two options are allowed for applying 3DGS optimization on Dl3DV-10K dataset with camera parameters and SfM points initialized as above.
   - Option 1: We first normalize camera parameters (centers/translation only) and SfM points into a unit (or a predefined radius `target_radius` in the code) sphere, and then run 3DGS optimization afterwards. 
   - Option 2: Or, we can run 3DGS optimization first, and then normalize camera parameters (centers/translation only) and the optimized 3D Gaussians into a unit (or a predefined radius `target_radius` in the code) sphere as a post-processing by normalizing their positions and anisotropic scaling factors. 
  Please refer to `sfm_camera_norm.py` for the implementation of normalization. Additionally, please refer to our `train.py` and related scripts for 3DGS optimization, which ensure that the output filenames match the corresponding input scenes from the DL3DV-10K dataset.
3. (optional) After normalizating camera parameters and 3D Gaussians, we can optionally run Semantics-aware filtering with [lang_sam](https://github.com/luca-medeiros/lang-segment-anything) to filter out the 3D Gaussians that are not relevant to the main objects of interest in the scene. Please refer to `groundedSAM.py` for the implementation of semantics-aware filtering.
4. Finally, we can run Can3Tok training and testing with 3D Gaussians (optionally filtered) as input. Please refer to `gs_can3tok.py` for the implementation.

## Cloning the Repository
```bash
git clone https://github.com/Zerg-Overmind/Can3Tok.git 
cd Can3Tok
```

## Environment Installation
We provide a conda environment file for easy installation. Please run the following command to create the environment:
```bash 
bash env_in_one_shot.sh
```
and then activate it:
```bash
conda activate can3tok
```
Please refer to the official repo to install [lang-sam](https://github.com/luca-medeiros/lang-segment-anything) for implementing our Semantics-aware filtering as in `groundedSAM.py`. Note that the pytorch version compatible with the latest lang-sam is `torch==2.4.1+cu121` instead of `torch==2.1.0+cu121` in our `env_in_one_shot.sh`, please modify the environment file accordingly if you want to use the latest lang-sam.
##


## Training and Testing
To train Can3Tok, please run the following command:
```bash
python gs_can3tok.py
```
where you might want to modify the path pointing to the 3D Gaussians path and output path in the script.
##


## Citation
If you find our code or paper useful, please consider citing:
```
@INPROCEEDINGS{gao2023iCCV,
  author = {Quankai Gao and Iliyan Georgiev and Tuanfeng Y. Wang and Krishna Kumar Singh and Ulrich Neumann and Jae Shin Yoon},
  title = {Can3Tok: Canonical 3D Tokenization and Latent Modeling of Scene-Level 3D Gaussians},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year = {2025}
}
```