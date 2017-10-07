# DNA Sequencing

## Overview

We classify DNA sequences from animals on a Fresh Polynesian island as invasive or native. Our dataset consists of about 1000 labeled DNA sequences, and 5000 unlabeled sequences. We attempt to build a robust binary classifer to determine if a DNA sequence is native or invasive.

We also look into regressing on a set of features of the DNA samples from examining the DNA sequence.

The goal of the project is to bactrace the introduction of invasive and nonnative speices to the ecosystem where the DNA samples were collected.

## Current Results

### Jeremy's baseline
Jeremy ran linear classification with hand labeled features.

### Utkarsh's methods

Utkarsh ran tests with a 10%/90% training/validation data split. 

Linear classification with features: 100% accuracy on validiation data, overfitting?

Clustering algorithm / semi-supervised learning: between 60-90% accuracy depending on hyperparameters.

