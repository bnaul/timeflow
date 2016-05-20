# Deep Learning for Unevenly-Spaced Time series
* Non-uniformly spaced?

## Outline
* Existing deep learning approaches for time series  
  * Speech/audio processing
  * Otherwise mostly forecasting, esp. financial
  * All assume uniformly-spaced
* Examples of non-uniform sampling
  * Astronomy / light curves
  * Definitely some bio{logy,informatics} use cases but not sure any would have enough data
  * Censored data maybe? i.e., uniform but with gaps
* Methods
  * Overview: autoencoder for bootstrapping w/ unlabeled
  (/augmented/even artificial) data, then fine-tune for classification
   * Input -> sequence-processing layers -> fixed-length encoding -> secondary input of times
(or lags) -> repeat encoding (sequence-length) times -> mirror of sequence-processing layers
-> output
* Layer types
  * LSTM (standard)
  * GRU (seems similar but trendy at the moment)
  * iRNN (vanilla RNN w/ clever initialization)
    * There are conflicting claims about whether they can model long-term dependencies or not but they're very easy to set up/train so it's worth a shot)
  * Convolutional
    * Good results in some time series problems but harder to tune since there
are so many hyperparameters
* Simulations
  * Test cases
    * Noisy sinusoidal data (superposition of many sinuosids?)
    * Periodic, non-sinusoidal?
    * Aperiodic?
    * Various noise levels, dataset sizes
  * Models
    * Various depths/types of layers
    * Model hyperparameters (just report best one for each architecture?)
* Applications
  * Survey classifier dataset?
  * Other light curves (not necessarily labeled)
