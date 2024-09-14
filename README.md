# Lunar Surface 3D Procedural Terrain Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Detailed Components](#detailed-components)
- [Current Progress](#current-progress)
- [Data Sources](#data-sources)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [License](#license)
- [Contact](#contact)

## Project Overview

This project aims to generate high-resolution 3D procedural terrain for lunar surfaces using advanced machine learning techniques. By combining Deep Convolutional Generative Adversarial Networks (DCGAN), Pix2Pix models, and GAN-based Digital Elevation Model (DEM) upscaling, we create detailed and accurate representations of lunar terrain based on data from the Chandrayaan 2 Orbiter's Terrain Mapping Camera.

The project is inspired by and builds upon the work presented in the paper:

> Zhang, F., Wu, B., Di, K., Liu, Z., Liu, Z., Liu, Y., & Ye, M. (2022). A Generative Adversarial Network for Pixel-Scale Lunar DEM Generation from High-Resolution Monocular Imagery and Low-Resolution DEM. Remote Sensing, 14(15), 3684. https://doi.org/10.3390/rs14153684

## Key Features

- [x] DCGAN-based heightmap generation
- [ ] Progressive growing for high-resolution (512x512) heightmaps
- [ ] Pix2Pix GAN for high-resolution terrain map generation from heightmaps
- [ ] DEM upscaling using GAN-based methods
- [ ] Integrated pipeline for end-to-end lunar terrain generation
- [ ] Evaluation metrics for generated terrain accuracy

## Detailed Components

### 1. DCGAN for Heightmap Generation

The DCGAN (Deep Convolutional Generative Adversarial Network) is used to generate lunar terrain heightmaps.

#### Architecture:
- Generator: Transposed convolutional layers with batch normalization and ReLU activation
- Discriminator: Convolutional layers with batch normalization and LeakyReLU activation

#### Key Features:
- [ ] Progressive growing implementation for high-resolution output
- [ ] Spectral normalization for improved training stability
- [ ] Custom loss function incorporating gradient penalty

### 2. Pix2Pix GAN for Terrain Map Generation

The Pix2Pix GAN transforms heightmaps into detailed, photorealistic lunar terrain images.

#### Architecture:
- Generator: U-Net architecture with skip connections
- Discriminator: PatchGAN for local and global feature assessment

#### Key Features:
- [ ] Custom data augmentation pipeline for lunar terrain
- [ ] Perceptual loss using pre-trained VGG network
- [ ] Multi-scale discriminator for improved global coherence

### 3. DEM Upscaling

GAN-based upscaling technique to generate high-resolution DEMs from low-resolution input and high-resolution imagery.

#### Architecture:
- Based on the approach described in Zhang et al. (2022)
- Modified U-Net generator with residual blocks
- PatchGAN discriminator similar to Pix2Pix

#### Key Features:
- [ ] Integration of low-resolution DEM and high-resolution image data
- [ ] Custom loss function combining adversarial, L1, and gradient difference losses
- [ ] Evaluation metrics specific to DEM quality (RMSE, slope accuracy)

### 4. Integrated Pipeline

A comprehensive system that combines all components for end-to-end lunar terrain generation.

#### Key Features:
- [ ] Automated workflow from input data to final terrain model
- [ ] User-friendly interface for parameter adjustment
- [ ] Visualization tools for generated terrains
- [ ] Export options for various 3D modeling formats

## Current Progress

- [x] Implemented DCGAN for 64x64 heightmap generation
- [x] Implemented DCGAN for 128x128 heightmap generation
- [ ] Progressive growing implementation (in progress)
- [ ] Pix2Pix GAN implementation
- [ ] DEM upscaling module
- [ ] Integrated pipeline development

```

## Data Sources

- Heightmaps extracted from DTM (Digital Terrain Model) of the Terrain Mapping Camera onboard the Chandrayaan 2 Orbiter
- High-resolution terrain maps/images from the Lunar Reconnaissance Orbiter Camera (LROC)
- 5m resolution DEMs (Digital Elevation Models) from LROC
- 0.2m resolution images of lunar terrain from LROC Narrow Angle Camera (NAC)

## Technologies Used

- Python 3.8+
- PyTorch 1.9+
- NumPy
- SciPy
- GDAL for geospatial data processing
- Matplotlib and Seaborn for visualization
- OpenCV for image processing
- MLflow for experiment tracking

## Contributing

We welcome contributions to this project. Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Acknowledgments

- Chandrayaan 2 mission and the Indian Space Research Organisation (ISRO) for providing valuable lunar data
- NASA's Lunar Reconnaissance Orbiter mission for high-resolution lunar imagery and DEMs
- The authors of "A Generative Adversarial Network for Pixel-Scale Lunar DEM Generation from High-Resolution Monocular Imagery and Low-Resolution DEM" for their innovative approach to DEM upscaling

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contact

Your Name - [@your_twitter](https://twitter.com/your_twitter) - email@example.com

Project Link: [https://github.com/yourusername/lunar-terrain-generation](https://github.com/yourusername/lunar-terrain-generation)
