# Automated Spray Control Based on Severity Level for Tomato Plants using Deep Learning and Image Processing

## Overview
The rapid and accurate identification of plant diseases is crucial for maintaining crop health and agricultural productivity. Traditional methods are often manual, time-consuming, and prone to errors. In recent years, advancements in machine vision technology, particularly leveraging deep learning and image processing techniques, have shown promise in addressing these challenges.

This repository contains code and resources for an end-to-end system designed to automate the detection and severity estimation of diseases in tomato plants. Our approach combines state-of-the-art deep learning models with image processing algorithms to analyze digital images of tomato leaves. This allows for not only detecting the presence of disease but also quantifying its severity.

## Key Features
- **Automated Disease Detection:** Utilizes deep learning models for accurate identification of disease presence in tomato plants.
- **Severity Estimation:** Provides quantitative assessment of disease severity, crucial for effective disease management strategies.
- **Integration with Agricultural Practices:** Designed to be integrated into existing agricultural workflows, facilitating informed decision-making and optimized treatment plans.
- **Cost Savings:** Enables significant reduction in pesticide usage by precisely measuring disease severity and implementing targeted treatment strategies.

## Repository Structure
- **`Disease Detection/`**: Contains dataset for training and implementation of the deep learning models for Disease Detection in Tomato Plants.
- **`Segmentation/`**: Contains code and dataset for image segmentation related to disease detection.
- **`Severity Estimation/`**: Includes algorithms and resources for quantifying disease severity from detected regions.
- **`Validation/`**: Stores the output and results from experiments and validation tests.

## Getting Started
To get started with using or contributing to this repository, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Neural-Ninja/Crop-Disease-Detection-and-Severity-Estimation-PLOS-ONE.git
   cd Crop-Disease-Detection-and-Severity-Estimation-PLOS-ONE
   
2. **Set up the environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

3. **Run a workflow from the repository root:**
   Run classifier scripts one at a time; each trains on the train split, validates on the validation split, evaluates on the test split, and writes its model/results beside the script. To train on another dataset, provide `--dataset-dir`; it must contain `train`, `val`, and `test` folders with one class subfolder per category. For example, a short full-dataset run is:
   ```powershell
      $dataset = ".\Disease Detection\dataset\Groundnut_Leaf_dataset_split"
      python ".\Disease Detection\Models\Inception-V3.py" --dataset-dir $dataset --epochs 5 --full-epoch
      python ".\Disease Detection\Models\Modified-Inception-V3.py" --dataset-dir $dataset --epochs 5 --full-epoch
      python ".\Disease Detection\Models\Resnet50.py" --dataset-dir $dataset --epochs 5 --full-epoch
      python ".\Disease Detection\Models\Modified-ViT.py" --dataset-dir $dataset --epochs 5 --batch-size 16
      ```
      Full training can take a long time on CPU.

   To run image-processing severity analysis and clustering:
   ```powershell
   python ".\Severity Estimation\Codes\severity_index.py"
   python ".\Validation\Clustering Approach\Codes\clustering_scaleogram.py"
   ```
   Use `--limit 5` with `severity_index.py` for a quick check. Full classifier training can take a long time on CPU. Video segmentation scripts require a video file, and scalogram generation requires a source-image directory; provide those inputs before running them.

4. **Explore the documentation:**
- Refer to `Docs/` for detailed documentation on the algorithms used, data formats, and experiment results.

## How to Contribute
Contributions to improve the functionality or performance of this system are welcome. Here are some ways you can contribute:
- Implement new features or improvements.
- Test the system with different datasets or scenarios.
- Report bugs or issues you encounter.
- Provide feedback on usability and documentation.

Please follow our [contribution guidelines](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md) when contributing to this project.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Note
This repository is made for our research publication under PLOS-One Journal.
