# MLops_project

MLops project
## Project Description
In this project, we aim to classify scenes using a Convolutional Neural Network(CNN) on the Places365 dataset (http://places2.csail.mit.edu/index.html), while maintaining a well-structured and documented code to allow for easy reproduction of results, leading to more trustworthy machine learning. Likewise, this project serves as a learning opportunity for this group to learn how to properly integrate a Machine Learning Operations(MLOps) flow, and its benefits.

We implement the CNN using the PyTorch Lightning high-level framework to reduce the boilerplate code and enable easier development of the Neural Network(NN) architecture. 
To further ensure the reproducibility of our results, we include the full specs for the experiment setup, including the code, python package versions, dockerfiles for the full experiment setup, as well as a Makefile to easily allow for replication. Likewise, relevant data from training the CNN will be made available as Weights&Biases reports, where we will log and visualize the progress and performance of different models.


The Places365 dataset is a large image dataset with more than 10 million images across 400+ different scene categories, with 5,000 to 30,000 images per category. The images all have a single label specifying their scene and allowing for simple classification, and as such allowing focus on the MLOps elements of the project.

The CNN model takes a starting point in the popular architectures [ResNet](https://github.com/KaimingHe/deep-residual-networks) and [VGG](https://gitlab.com/vgg/vgg_classifier) and potentially modifies the models for enhanced performance or other alterations. While not initially in the project scope to focus on groundbreaking developments in the performance of classification, the project does intend to determine if optimizations can be done to the runtime of both training and inference, by investigating the runtime profiles, and/or comparing the runtimes of the 2 listed CNN models.

To comply with common Python coding standards, this project attempts to adhere to PEP8 formatting as well as utilizing strict typing and integrating relevant unittesting to allow for easier development of new features throughout the project.


## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── MLops_project  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
