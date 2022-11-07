## Usage
------------------------------

In `/scripts` you may find some scripts prepared to run the default values with the only input being the dataset to be used, through the argument --dataset.

*NOTE:* Configure the paths to the datasets by editting the file in qdf/settings.py:
```python
DATASET_PATH = ".../QuantumDeepField_molecule/dataset"
SAVE_PATH = ".../QuantumDeepField_molecule/output"
```
----------------------------------------
### 1. Preprocessing (for training):

```bash
 python preprocess_train.py --dataset=$dataset_trained
```

e.g _python preprocess_train.py --dataset=QM9under7atoms_homolumo_eV_

#### Options:

* `dataset` [required]: [string] dataset to be used in pre-training. From those that can be installed directly from the cloned repository the options are:
    * "QM9under14atoms_atomizationenergy_eV"
    * "QM9full_atomizationenergy_eV"
    * "QM9full_homolumo_eV" Note: Two properties (homo and lumo)
    * "<your choice>"

----------------------------------------
### 2. Training:

```shell
 python train.py --dataset=$dataset_trained --num_workers=$num_workers --seed=$seed --device=$device
```

e.g _python train.py --dataset=QM9under7atoms_homolumo_eV_

#### Options:

 * `dataset` [required]: [string] dataset to be used in pre-training. From those that can be installed directly from the cloned repository the options are:
    * "QM9under14atoms_atomizationenergy_eV"
    * "QM9full_atomizationenergy_eV"
    * "QM9full_homolumo_eV" Note: Two properties (homo and lumo)
    * "<your choice>"
 * `num_workers`: [int] number of workers to use for the dataloader. Defaults to 1.
 * `seed`: [int] integer used to specify the seed for the model initialization. Defaults to 1729.
 * `device`: [string] device to use for training and inference in the model, options are ["cuda", "cpu"], if None is specified it will use "cuda" if available in your system, else will use "cpu" (slower).
 
---------------------------------------------------
### 3. Preprocessing inference (predict):

```shell
 python preprocess_predict.py --dataset_train=$dataset_trained --dataset_predict=$dataset_predict
```
e.g python preprocess_predict.py --dataset_train=QM9under7atoms_homolumo_eV --dataset_predict=QM9full_homolumo_eV

#### Options:

 * `dataset_train` [required]: [string] dataset that was used in pre-training. It is use to look for and load the appropriate orbital dictionaries so that the preprocessing done in the prediction dataset is coherent to what was done in pre-processing the original dataset trained on.
 * `dataset_predict` [required]: [string] dataset to be used in prediction.


----------------------------------------
### 4. Prediction (Inference):

```shell
 python predict.py --dataset_train=$dataset_trained --dataset_predict=$dataset_predict --model_path=$model_path --num_workers=$num_workers --seed=$seed --device=$device
```
e.g python predict.py --dataset_train=QM9under7atoms_homolumo_eV --dataset_predict=QM9full_homolumo_eV --model_path="../pretrained/model"

#### Options:

 * `dataset_train` [required]: [string] dataset that was used in pre-training. It is use to look for and load the appropriate orbital dictionaries so that the preprocessing done in the prediction dataset is coherent to what was done in pre-processing the original dataset trained on.
 * `dataset_predict` [required]: [string] dataset to be used in prediction.
 * `model_path` [required]: [string] path to file where the pre-trained model is saved.
 * `num_workers`: [int] number of workers to use for the dataloader. Defaults to 1.
 * `seed`: [int] integer used to specify the seed for the model initialization. Defaults to 1729.
 * `device`: [string] device to use for training and inference in the model, options are ["cuda", "cpu"], if None is specified it will use "cuda" if available in your system, else will use "cpu" (slower).