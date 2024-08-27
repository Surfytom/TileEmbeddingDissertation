# Title: WIP

This repo is a forked version of the [tile embedding repository](https://github.com/js-mrunal/tile_embeddings) which is created by one of the authors (Mrunal Jadhav) of the paper it partly implements [tile embedding paper](https://ojs.aaai.org/index.php/AIIDE/article/view/18888).

This repository re-implements the aforementioned tile embedding paper in it's entirety using the authors repository linked above as a sanity checker. This repository also proposes adding attention into the orginal author's autoencoder architecture to **potentially improve** performance.

This repository utilises the PyTorch library to implement the models shown in the paper (the original repo used Keras).

## Contents
1. [Setup Repository and Enviroment](#setup-repository)
2. [How To Use the Repository]()

## Setup Repository

This section will show you how to setup the repository and configure your enviroment to be able to run the notebooks in the repository.

### 1 | Clone the Github Repository
Run this command in your preferred terminal to download the repository:

```
git clone https://github.com/Surfytom/TileEmbeddingDissertation.git
```

### 2 | Setting Up the Enviroment

#### Option 1 - Conda

This option requires miniconda or anaconda to be installed already.

There is a enviroment.yaml file in the repository. Navigate your terminal to the base path of the repository then run the following command:

```
conda env create -f environment.yml
```
This conda enviroment can then be used in the notebooks by selecting the enviroment in the top right of a notebook with the option **"Select Kernel"**

#### Option 2 - Pip

There is a requirments.txt file in the base folder of the repository that can be used to install the repositories packages using pip. First navigate to the base folder of the repository then run this command:

```
pip install -r requirements.txt
```
This pip enviroment can then be used in the notebooks by selecting the enviroment in the top right of a notebook with the option **"Select Kernel"**

## How To Use the Repository



## 1 Data Extraction and Preparation
**Prerequisites:**
* Step 0

The training data for our implementation includes five games: *Super Mario Bros, Kid Icarus, Legend of Zelda, Lode Runner, Megaman*. To train the autoencoder for obtain an embedded representation of tile, we draw on local pixel context and the affordances of the candidate tile. 

> 1. Local Pixel Context: To extract the 16 * 16 tiles along with its local context, we slide a 48 * 48 window over the level images. The parent dataset for level images is [VGLC](https://github.com/TheVGLC/TheVGLC). However, level images for some games have extra pixels along the vertical/horizontal axis which result in off-centered tile sprite extraction(demonstrated in fig). We perform prelimnary image manipulations on this dataset to fit the dimensions of such level images. Lode Runner levels has 8 * 8 tile size which we upscaled to 16 * 16 using the [PIL](https://pillow.readthedocs.io/en/stable/) library. We provide the preprocessed dataset directory [vglc](https://github.com/js-mrunal/tile_embeddings/tree/main/data/vglc).

<img src="images/data_extraction.png">

To extract the context for all five games run the following. 
1. Move to directory: context_extraction
```
cd notebooks/
```

2. Run the following command in shell
```
pipenv run python extract_context_data.py
```

On successful execution of this code, navigate to the folder *data > context_data >*. Each game directory populated with visual local context seperated with sub-directories of tile characters. Each game directory also has an JSON file created. It is a dictionary with key as the centre tile, enlisting all possible neighbourhoods it. 

> 2. Affordances: 
We define a single, unified set of 13 tags across the games. The tile character to behaviour mapping is provided as [JSON](https://github.com/js-mrunal/tile_embeddings/tree/main/data/json_files_trimmed_features) files. 
    
    
Inputs obtained are as follows: 

<img src="images/inputs.png">

## 2 Autoencoder for Tile Embeddings
**Prerequisites:**
* Step 0
* You can skip Step 1 ONLY IF you want to directly load the provided autoencoder architecture and its pretrained weights (Step (2c)).

Now that we have both our inputs ready, we have to integrate them into a single latent vector representation, which we call *tile embeddings*. To get the embedding, we employ the X-shaped autoencoder architecture. Please find the details of the architecture in our paper. 

2a. The jupyter notebook "notebooks > autoencoder_training.ipynb" provides a step by step guide for autoencoder training. 

2b. You can also run the following commands to train the autoencoder and save the weights:

> Move to the directory: notebooks
```
cd notebooks/
```
> Run the following command in shell
```
pipenv run python autoencoder_training.py
```
2c. Load the directly provided architecture and pretrained weights to perform evaluation. Sample Notebook:  
```
evaluating_autoencoder_on_text.ipynb
```

## 3 Unified Level Representation with Tile Embeddings
**Prerequisites:**
* Step 0
* (Optional) Step 1 followed by Step 2 

In this step we convert the levels to the embedding representation using tile embeddings. There are two ways to go about this step. 

3a. If the game data has affordance mapping present, leverage it to get the tile embeddings. The following notebook converts the  levels of all the games in our training dataset into a unified level representation by considering visuals as well as affordances. 
```
pipenv run python generate_unified_rep.py
```
After the execution of this notebook the embedding representation of the levels for the games *Super Mario Bros, Kid Icarus, Megaman, Legend of Zelda, Lode Runner* will be stored in the directory *data>unified_rep*. 

3b. In case of missing affordances, we provide evidence that the autoencoder can still approximate the embedding vectors (refer paper). To address such datasets, refer the following script
```
pipenv run python rep_no_affordances.py
```
After the execution of this python code the embedding representation of the *Bubble Bobble* levels will be stored in *data>unified_rep>* 

*Note: Feel free to skip this step and directly use the provided the pickle files. Navigate to data>unified_rep, you will find the embedding representation of the levels for the games: Super Mario Bros, Kid Icarus, Megaman, Legend of Zelda, Lode Runner, Bubble Bobble *

## 4 Generating Level Representation for Bubble Bobble
**Prerequisites:**

The notebook *bubble_bobble_generation.ipynb* provides step-by-step instructions in detail for generating levels of the game Bubble Bobble using LSTM and tile embeddings.
 

## How can you contribute to this project?

* Add more data! Training the autoencoder on more games and enriching the corpus 
* Perform cool experiments with tile embedding, and cite us :)
* Find bugs and help us make this repository better!
