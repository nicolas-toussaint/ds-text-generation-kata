Text generation using an $n$-gram language model
=====

This repository contains the kata result.

# How to use this repository

## Install

You can install this project using the poetry tool. 

```bash
poerty install
```

## Usage

To train the model, you can use the following command.
You have to give a training config path with the model parameters.

```bash
$ python -m TextGeneration.train TRAINING_CONFIG_PATH
```

To generate texts using the trained model, you can use the following command.
You have to give an input config path with the model parameters and can set a logger level.

```bash
$ python -m TextGeneration.generate INPUT_CONFIG_PATH [LOGGER_LEVEL]
```
