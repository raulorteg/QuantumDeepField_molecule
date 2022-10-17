Note: This is a fork of the original implementation (https://github.com/masashitsubaki/molecularGNN_smiles)

# Quantum deep field for molecule

<div align='center'>
<p><img src='figure/logo.jpeg' width='1000'/></p>
</div>

<div align='center'>
<p><img src='figure/overview.jpeg' width='1000'/></p>
</div>

## Installation 
------------------------------
1. Clone the repository: ```git clone https://github.com/raulorteg/QuantumDeepField_molecule```
2. Create the python virtual environment (I use python 3.9.14): ```virtualenv --py=python3.9 qdf```
3. Activate virtualenv  ```source qdf/bin/activate```
4. Install requirements ```python -m pip install -r requirements.txt``` Note: Your system might need a different torch installation (https://pytorch.org/get-started/locally/)

## Requirements
------------------------------

see the ```requirements.txt``` file


## Usage
------------------------------

In `/scripts` you may find some scripts prepared to run the default values with the only input being the dataset to be used, through the argument --dataset.

*NOTE:* Configure the paths to the datasets by editting the file in qdf/settings.py:
```python
DATASET_PATH = "/home/raul/git/QuantumDeepField_molecule/dataset"
SAVE_PATH = "/home/raul/git/QuantumDeepField_molecule/output"
```

### 1. Preprocessing (for training):

```shell
 python preprocess_train.py --dataset=$dataset_trained
 ```
 _e.g `python preprocess_train.py --dataset=QM9under7atoms_homolumo_eV`_

 _Options:_
 * `dataset`: [string] dataset to be used in pre-training. From those that can be installed directly from the cloned repository the options are:
    * "QM9under14atoms_atomizationenergy_eV"
    * "QM9full_atomizationenergy_eV"
    * "QM9full_homolumo_eV" Note: Two properties (homo and lumo)
    * "<your choice>"

### 2. Training:

```shell
 python train.py --dataset=$dataset_trained --num_workers=$num_workers --seed=$seed --device=$device
 ```
 _e.g `python train.py --dataset=QM9under7atoms_homolumo_eV`_

 _Options:_
 * `dataset` [required]: [string] dataset to be used in pre-training. From those that can be installed directly from the cloned repository the options are:
    * "QM9under14atoms_atomizationenergy_eV"
    * "QM9full_atomizationenergy_eV"
    * "QM9full_homolumo_eV" Note: Two properties (homo and lumo)
    * "<your choice>"
 * `num_workers`: [int] number of workers to use for the dataloader. Defaults to 1.
 * `seed`: [int] integer used to specify the seed for the model initialization. Defaults to 1729.
 * `device`: [string] device to use for training and inference in the model, options are ["cuda", "cpu"], if None is specified it will use "cuda" if available in your system, else will use "cpu" (slower).
 
### 3. Preprocessing inference (predict):

```shell
 python preprocess_predict.py --dataset_train=$dataset_trained --dataset_predict=$dataset_predict
 ```
 _e.g `python preprocess_predict.py --dataset_train=QM9under7atoms_homolumo_eV                        --dataset_predict=QM9full_homolumo_eV`
 _Options:_
 * `dataset_train` [required]: [string] dataset that was used in pre-training. It is use to look for and load the appropriate orbital dictionaries so that the preprocessing done in the prediction dataset is coherent to what was done in pre-processing the original dataset trained on.
 * `dataset_predict` [required]: [string] dataset to be used in prediction.

### 4. Prediction (Inference):

```shell
 python predict.py --dataset_train=$dataset_trained --dataset_predict=$dataset_predict --model_path=$model_path --num_workers=$num_workers --seed=$seed --device=$device
 ```
 _e.g `python predict.py --dataset_train=QM9under7atoms_homolumo_eV                        --dataset_predict=QM9full_homolumo_eV --model_path="../pretrained/model"`

 _Options:_
 * `dataset_train` [required]: [string] dataset that was used in pre-training. It is use to look for and load the appropriate orbital dictionaries so that the preprocessing done in the prediction dataset is coherent to what was done in pre-processing the original dataset trained on.
 * `dataset_predict` [required]: [string] dataset to be used in prediction.
 * `model_path` [required]: [string] path to file where the pre-trained model is saved.
 * `num_workers`: [int] number of workers to use for the dataloader. Defaults to 1.
 * `seed`: [int] integer used to specify the seed for the model initialization. Defaults to 1729.
 * `device`: [string] device to use for training and inference in the model, options are ["cuda", "cpu"], if None is specified it will use "cuda" if available in your system, else will use "cpu" (slower).


## Datasets
------------------------------

The QM9full dataset provided in this repository contains 130832 samples;
the [original QM9 dataset](https://springernature.figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904)
contains 133885 samples but we removed 3053 samples that
[failed the consistency check](https://springernature.figshare.com/articles/dataset/Uncharacterized%3A_List_of_3054_molecules_which_failed_the_geometry_consistency_check/1057644)
(i.e., 130832 = 133885 - 3053).

We note that, as described in [README of the QM9 dataset](https://springernature.figshare.com/articles/dataset/Readme_file%3A_Data_description_for__Quantum_chemistry_structures_and_properties_of_134_kilo_molecules_/1057641),
the original QM9 dataset provides U<sub>0</sub>
as the internal energy at 0 K in units of Hartree.
We transformed the internal energy into the atomization energy E in units of eV,
which can be calculated using [Atomref of the QM9 dataset](https://springernature.figshare.com/articles/dataset/Atomref%3A_Reference_thermochemical_energies_of_H%2C_C%2C_N%2C_O%2C_F_atoms./1057643),
that is, E = U<sub>0</sub> - sum(Atomrefs in the molecule), and 1 Hartree = 27.2114 eV.

In this way, we created the atomization energy dataset,
extracted the QM9under14atoms and QM9over15atoms datasets from it,
and provided them in the dataset directory (note that
the QM9over15atoms contains only test.txt for extrapolation evaluation).
On the other hand, the homolumo dataset does not require such preprocessing
and we only transformed their units from Hartree into eV.
The final format of the preprocessed QM9 dataset is as follows.

<div align='center'>
<p><img src='figure/dataset.jpeg' width='1000'/></p>
</div>

Note that our QDF model can learn multiple properties simultaneously
(i.e., the model output has multiple dimensions)
when the training dataset format is prepared as the same as the above QM9full_homolumo_eV.
