## Installation Steps

1. **Create Conda Environment**:
    ```bash
    conda create -f env.yaml
    ```
    This will build the Conda environment.

2. **Activate Conda Environment**:
    ```bash
    conda activate ssl-feats
    ```

3. **Clone SimCLRv2-Pytorch Repository**:
    ```bash
    git clone https://github.com/nmehlman/SimCLRv2-Pytorch
    ```
    This is needed for loading the supervised models from the SimCLRv2 repo.

4. **Set Environment Variables**:
    - `MODEL_WEIGHTS_DIR`: Path where model weights are stored and loaded.
    - `SIM_CLR_PYTORCH_DIR`: SimCLRv2-Pytorch repo directory.

5. **Add This Repository to Your Python Path**:
    ```bash
    export PYTHONPATH=$PYTHONPATH:/path/to/this/repo
    ```

6. **Download Weights Files**:
    You can download weights files from the MMSelfSup repo using the `download_and_parse_mmselfsup_weights.py` file. Refer to the [MMSelfSup Model Zoo](https://mmselfsup.readthedocs.io/en/latest/model_zoo.html) for weights files.