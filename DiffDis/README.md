# [DiffDis: Empowering Generative Diffusion Model with Cross-Modal Discrimination Capability](https://openaccess.thecvf.com/content/ICCV2023/papers/Huang_DiffDis_Empowering_Generative_Diffusion_Model_with_Cross-Modal_Discrimination_Capability_ICCV_2023_paper.pdf)

Runhui Huang, Jianhua Han, Guansong Lu, Xiaodan Liang, Yihan Zeng, Wei Zhang, and Hang Xu

*ICCV 2023*

<div align="center">
    <img src="code/images/sample_image.png" alt="Sample Image">
    <p><em>Sampled image using the prompt: A large bed sitting next to a small Christmas Tree surrounded by pictures</em></p>
</div>

This folder provides a re-implementation of this paper in PyTorch, developed as part of the course METU CENG 796 - Deep Generative Models. The re-implementation is provided by:
* Furkan Genç, genc.furkan@metu.edu.tr 
* Barış Sarper Tezcan, baris.tezcan@metu.edu.tr

Please see the jupyter notebook file [main.ipynb](code/main.ipynb) for a summary of paper, the implementation notes and our experimental results


# Project Setup

### Setup Environment

To set up the environment for this project, please follow the steps below:

1. **Create a new conda environment** named `DiffDis`:
    ```bash
    conda create -n DiffDis
    ```

2. **Activate** the newly created environment:
    ```bash
    conda activate DiffDis
    ```

3. **Install PyTorch** and related libraries with CUDA support:
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

4. **Install additional required Python packages** from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

### Change Directory

Navigate to the `scripts` directory where the project scripts are located:
```bash
cd scripts
```

### Download Data

To download the necessary data for the project, execute the following script:
```bash
bash download_data.sh
```

### Download and Extract CC3M Dataset

To download and extract the CC3M dataset, run the following scripts in order:

1. **Download the CC3M dataset**:
    ```bash
    bash download_cc3m_dataset.sh
    ```

2. **Extract the downloaded CC3M dataset**:
    ```bash
    bash extract_cc3m_dataset.sh
    ```

# Usage

### Change Directory

Navigate to the `code` directory where the project code is located:
```bash
cd code
```

1. **Training**: Run the following command to train the model:
    ```bash
    python train.py
    ```

2. **Testing**: Run the following command to test the model:
    ```bash
    python test.py
    ```

3. **Inference**: Run the following command to generate images:
    ```bash
    python inference.py
    ```

**Note**: Adjust the `config.py` parameters according to your needs before running the scripts.

### Pre-trained Models

The model trained on the CC3M dataset for two-thirds of an epoch can be found in this [link](https://drive.google.com/uc?id=1iVTS0fYkmKkT4s5EcZoMC9L8-p29dW41).