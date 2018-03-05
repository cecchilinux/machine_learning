# Traffic sign classifier
Machine learning course project. We build a traffic sign classifier with multi-scale Convolutional Networks using Keras starting from [this](http://ieeexplore.ieee.org/document/6033589/) publication, obtaining the result of 99.33% in the [GTSRB dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) version online-competition (the same used on the publication) improving the accuracy on test-set.

You can find all the details on this Jupiter [notebook](notebooks/report.ipynb) written in Italian :'( .

## Requirements
If you want to use Docker read the [docker/README.md](docker), otherwise you need on your system:
- python3
- jupyter
- numpy
- pandas
- scikit-learn
- matplotlib
- scikit-image
- glob2
- opencv
- keras

## Usage

```bash
cd scripts
python preprocessing.py --help
python training.py --help
python test_on_testset.py --help
python test_on_new.py --help

python training.py --augmentation --blur --epochs 40 -- dataset online
```
