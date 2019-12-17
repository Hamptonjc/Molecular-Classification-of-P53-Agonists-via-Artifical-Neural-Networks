# Molecular Classification of P53 Agonists via Artifical Neural Networks

This was an undergraduate research project conducted by [Jonathan Hampton](www.linkedin.com/in/hamptonjc) in November of 2019 at East Tennessee State University. The project advisor was [Dr. Jeff Knisley](https://sites.google.com/site/drjknisley/).

## Description

Deep learning algorithms have been applied to problems in a variety of fields. In this project they were applied to the scientific discipline of Toxicology. The data that was used came from the Tox21 data set; a collection of over 10,000 different chemicals and drugs along with the results of 12 different toxicity-indicating assays. For this project only the stress-response p53 assay was used. This resulted in an imbalanced data set of 6,351 non-toxic molecules and only 423 toxic molecules. To combat this, several types of data re-sampling methods were tried and their results were compared. Two artificial neural networks were built using the popular machine learning library Pytorch. The networks were trained independently and the performance of their different architectures were compared. The best performing model had a top accuracy of 93% as well as a precision of 42% on toxic samples and 96% on non-toxic samples. A different instance of that model, one of which was trained with a different data re-sampling method, had the largest area under the ROC curve at 0.81.

## Requirements

* deepchem                  2.3.0

* rdkit                     2019.03.4.0

* pytorch                   1.3.0

* scikit-learn              0.21.3

* skorch                    0.6.0

* imbalanced-learn          0.5.0

* pandas                    0.24.2

* numpy                     1.17.3

* matplotlib                3.1.1

* ipython                   7.8.0
