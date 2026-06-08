

# REGEN: Real-Time Photorealism Enhancement in Games via a Dual-Stage Generative Network Framework

<div style="display: flex; justify-content: center; gap: 10px;">
  <a href="https://ieeexplore.ieee.org/document/11373202" target="_blank">
    <img alt="Static Badge" src="https://img.shields.io/badge/PAPER-blue?style=for-the-badge&logo=IEEE&logoSize=auto">
  </a>
  <a href="https://www.youtube.com/watch?v=tEgk4ycpmpQ" target="_blank">
    <img alt="Static Badge" src="https://img.shields.io/badge/DEMO-red?style=for-the-badge&logo=YouTube&logoSize=auto">
  </a>
  <a href="https://drive.google.com/drive/folders/19Q8E9wy3MR-vUfOfVwytIzv4QNpndYo0" target="_blank">
<img alt="Static Badge" src="https://img.shields.io/badge/MODELS-orange?style=for-the-badge&logo=googledrive&logoSize=auto">
  </a>
    <a href="https://github.com/stefanos50/CARLA2Real" target="_blank">
<img alt="Static Badge" src="https://img.shields.io/badge/CARLA2Real_Project-black?style=for-the-badge&logo=github&logoSize=auto">
  </a>
      <a href="https://github.com/stefanos50/HyPER-GAN" target="_blank">
<img alt="Static Badge" src="https://img.shields.io/badge/HyPERGAN_Project-black?style=for-the-badge&logo=github&logoSize=auto">
  </a>
</div>

## Demonstration

The following demo illustrates a side-by-side comparison of the framework performing `GTAV → Cityscapes` (left) and `GTAV → Mapillary Vistas` (right) at `1280x720 (maximum game settings)`. The footages were recorded with OBS Studio while the game was also rendered on the same GPU. It is running on a system with an `RTX 4090`, an `Intel i7 14700F CPU`, and `64GB of DDR4` system memory without any optimization (e.g., TensorRT). The full video is included in the `demos` directory.

<p align="center">
  <img src="./demos/gta2cs.gif" width="45%" />
  <img src="./demos/gta2vistas.gif" width="45%" />
</p>

## Diffusion Models

Below is a real-time demo (RTX 4070S) of REGEN trained to translate CARLA towards the output of FLUX.2-klein-4B. Due to the more frequent inconsistencies of diffusion models (e.g., changing the color of the vehicles) compared to image-to-image translation, there are more frequent instances of temporal instability (e.g., flickering) compared to traditional image-to-image translation methods. However, these issues can be mitigated by leveraging more advanced, paid diffusion-based models, such as ChatGPT Image or Qwen Image 2.0.


https://github.com/user-attachments/assets/288eceee-299f-473d-a144-386396bdc1b6


### Updates

* **09/06/2026**: Added code for exporting the models into ONNX format. Added sample code for inference through ONNX Runtime. Added instructions for integrading the models into Unreal Engine 5.
* **22/03/2026**: Added a model trained on the output of FLUX.2-klein-4B.
* **16/12/2025**: Added pretrained model for [nuScenes](https://www.nuscenes.org/).


## Abstract

Photorealism is an important aspect of modern video games since it can shape the player experience and simultaneously impact the immersion, narrative engagement, and visual fidelity. Although recent hardware technological breakthroughs, along with state-of-the-art rendering technologies, have significantly improved the visual realism of video games, achieving true photorealism in dynamic environments at real-time frame rates still remains a major challenge due to the tradeoff between visual quality and performance. In this short paper, we present a novel approach for enhancing the photorealism of rendered game frames using generative adversarial networks. To this end, we propose Real-time photorealism Enhancement in Games via a dual-stage gEnerative Network framework (REGEN), which employs a robust unpaired image-to-image translation model to produce semantically consistent photorealistic frames that transform the problem into a simpler paired image-to-image translation task. This enables training with a lightweight method that can achieve real-time inference time without compromising visual quality. We demonstrate the effectiveness of our framework on Grand Theft Auto V, showing that the approach achieves visual results comparable to the ones produced by the robust unpaired Im2Im method while improving inference speed by 32.14 times. Our findings also indicate that the results outperform the photorealism-enhanced frames produced by directly training a lightweight unpaired Im2Im translation method to translate the video game frames towards the visual characteristics of real-world images.

### BibTeX Citation

If you used the REGEN framwork or any of the pretrained models from this repository in a scientific publication, we would appreciate using the following citation:

```
@ARTICLE{11373202,
  author={Pasios, Stefanos and Nikolaidis, Nikos},
  journal={IEEE Transactions on Games}, 
  title={REGEN: Real-Time Photorealism Enhancement in Games Via a Dual-Stage Generative Network Framework}, 
  year={2026},
  volume={},
  number={},
  pages={1-8},
  keywords={Games;Translation;Photorealism;Visualization;Semantics;Video games;Real-time systems;Training;Engines;Rendering (computer graphics);Computer vision;image-to-image translation;photorealism enhancement;unreal engine},
  doi={10.1109/TG.2026.3661622}}
```

> 📝 **Note**: This repository uses code from the [Pix2PixHD repository](https://github.com/NVIDIA/pix2pixHD).

## Requirements

```
conda create -n REGEN python=3.9
conda activate REGEN
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install carla==0.9.15
pip install opencv-python
pip install dominate
pip install scipy
pip install mss
pip install pywin32
pip install onnxruntime-gpu
```

## Training

To train the model, it is required to have access to a synthetic dataset generated by a game or simulator and the corresponding images that were photorealism enhanced by a robust unpaired image-to-image translation method such as [Enhancing Photorealism Enhancement (EPE)](https://github.com/isl-org/PhotorealismEnhancement). 

### CARLA Simulator

To train a model that enhances the photorealism of the CARLA simulator towards the characteristics of real-world datasets (Mapillary Vistas, Cityscapes, and KITTI), we already provide both the original rendered frames and the results of EPE [here](https://www.kaggle.com/datasets/stefanospasios/carla2real-enhancing-the-photorealism-of-carla).

### Grand Theft Auto V

To train a model that enhances the photorealism of GTAV towards the characteristics of real-world datasets (Mapillary Vistas and Cityscapes), the results of EPE are already provided by the authors at the [official repository](https://github.com/isl-org/PhotorealismEnhancement). The initial rendered GTAV frames originated from the Playing for Data dataset, which can be downloaded [here](https://download.visinf.tu-darmstadt.de/data/from_games/).

### Starting the Training

After collecting the required datasets, place the training and test sets of the game/simulator dataset into `code/data/train_A` and `code/data/test_A`, respectively. The corresponding photorealism-enhanced images should be transferred into the `code/data/train_B` and `code/data/test_B` directories. To start training, execute the following command:

```javascript
python train.py --dataroot ./data --name REGEN --label_nc 0 --no_instance --gpu_id 0
```

## Testing

To test the framework, we provide pretrained models for `GTAV → Cityscapes`, `CARLA → Cityscapes`, and `CARLA → KITTI`. Download the models from [Google Drive](https://drive.google.com/drive/folders/19Q8E9wy3MR-vUfOfVwytIzv4QNpndYo0?usp=sharing) and transfer them into `code/checkpoints/REGEN/`. Finally, transfer the images that are to be inferred with the model in the `code/data/test_A` directory and execute the following command:

```javascript
python test.py --dataroot ./data --name REGEN --label_nc 0 --no_instance --gpu_id 0
```

The resulting images will be saved in the `code/results/REGEN/images/` directory.

> 📝 **Note**: We have already provided some sample screenshots for testing purposes that also include the UI of the game.

<img width="1216" height="337" alt="test_images" src="https://github.com/user-attachments/assets/eb74afeb-604e-4569-81bf-8b97a3c9cb20" />

## Real-Time Inference

We additionally provide two sample scripts for testing the models in real-time conditions. The provided pretrained models should be placed in the same directory as for testing.

### CARLA Simulator

To test the model on CARLA, download the UE4 executable of the simulator from the [official repository](https://github.com/carla-simulator/carla/releases). Particularly, the code was tested with CARLA version 0.9.15. After running the simulator and initializing the world, execute the following command:

```javascript
python carla_test.py --dataroot ./data --name REGEN --label_nc 0 --no_instance --gpu_id 0
```
<img width="1233" height="289" alt="carla" src="https://github.com/user-attachments/assets/72729705-0a54-43ff-b27d-0f0887f6122f" />

### Grand Theft Auto V

To test the model on GTA V, first download and run the game. Considering that the script performs real-time capturing of the game window, set the game in windowed mode with a lower resolution of the monitor (a dual-monitor setup would be ideal). In addition, through the game settings cap the frame rate to 30 FPS to reduce the GPU load. Then execute the following script:

```javascript
python gta_test.py --dataroot ./data --name REGEN --label_nc 0 --no_instance --gpu_id 0
```
> ⚠️ **Warning**: You may need to modify the offsets in line 60 of `gta_test.py` in order to perfectly crop the game window while capturing.

> 📝 **Note**: For the best results, it is recommended to download [ScriptHook](https://www.gta5-mods.com/tools/script-hook-v) and [Hood Camera](https://www.gta5-mods.com/scripts/hood-camera) mods, as the PFD dataset used for training is mainly limited to that perspective.

> 📝 **Note**: All the available parameters of the model (e.g., for changing the resolution of the resulting images) can be found in `code/options/`.

## Integration

In order to easily integrate the models into your own pipelines, we provide code for transforming the models into the widely used for deployment ONNX format. To export a model into ONNX, run the following command:

```javascript
%example command
python regen_onnx_eport.py --input <path-to>\carla2kitti.pth --output <path-to>\carla2kitti.onnx --height 544 --width 960
```

A sample script is also provided in `onnx_utils/test_onnx.py` to understand the preprocessing as well as the postprocessing steps that are required for inference with ONNX Runtime. To test the exported ONNX model on an image, run the following command:

```javascript
%example command
python test_onnx.py --onnx <path-to>/carla2kitti.onnx --image <path-to>/image.jpg --height 544 --width 960 --output <path-to>/output.jpg
```
### Unreal Engine 5 Integration

With the release of the Unreal Engine 5 version 5.4 and above the engine now supports the real-time integration of neural rendering models through ONNX runtime. The integration requires no more than 7-8 minutes following the video tutorial: [see the tutorial here](https://www.youtube.com/watch?v=OjdG4TqBozg). Below, we provide the exact preprocessing and postprocessing steps that should be applied to the post-processing material:

<img width="1259" height="759" alt="Screenshot 2026-06-09 003738" src="https://github.com/user-attachments/assets/f6d43c6e-d0d1-45ae-8681-ed0164039ba6" />

> 📝 Note: In Unreal Engine, the GPU will have to render both the engine's synthetic environment and run the model. At a resolution of 960x544, an RTX 4090 can achieve above 20 FPS when integrating REGEN into Unreal Engine 5.

