# Ensemble Adversarial Training

This repository contains code to reproduce results from the paper:

**Ensemble Adversarial Training: Attacks and Defenses** <br>
*Florian Tramèr, Alexey Kurakin, Nicolas Papernot, Dan Boneh and Patrick McDaniel* <br>
ArXiv report: https://arxiv.org/abs/1705.07204

<br>

###### REQUIREMENTS

The code was tested with Python 2.7.12, Tensorflow 1.0.1 and Keras 1.2.2.

###### EXPERIMENTS

We start by training a few simple MNIST models. These are described in _mnist.py_.

```
python -m train models/modelA --type=0
python -m train models/modelB --type=1
python -m train models/modelC --type=2
python -m train models/modelD --type=3
```

Then, we can use (standard) Adversarial Training or Ensemble Adversarial Training 
(we train for either 6 or 12 epochs in the paper). With Ensemble Adversarial 
Training, we additionally augment the training data with adversarial examples 
crafted from external pre-trained models (models A, C and D here):

```
python -m train_adv models/modelA_adv --type=0 --epochs=12
python -m train_adv models/modelA_ens models/modelA models/modelC models/modelD --type=0 --epochs=12
```

The accuracy of the models on the MNIST test set can be computed using

```
python -m simple_eval test [model(s)]
```

To evaluate robustness to various attacks, we use

```
python -m simple_eval [attack] [source_model] [target_model(s)] [--parameters (opt)]
```

The attack can be: 

| Attack | Description | Parameters |
| ------ | ----------- | ---------- |
| fgs    | Standard FGSM | *eps* (the norm of the perturbation) |
|rand_fgs| Our FGSM variant that prepends the gradient computation by a random step | *eps* (the norm of the total perturbation); *alpha* (the norm of the random perturbation) |
| ifgs   | The iterative FGSM | *eps* (the norm of the perturbation); *steps* (the number of iterative FGSM steps) |
| CW  | The Carlini and Wagner attack | *eps* (the norm of the perturbation); *kappa* (attack confidence) |

Note that due to GPU non-determinism, the obtained results may vary by a few 
percent compared to those reported in the paper.
Nevertheless, we consistently observe the following:

* Standard Adversarial Training performs worse on transferred FGSM 
  examples than on a "direct" FGSM attack on the model due to a *gradient masking* effect.
* Our RAND+FGSM attack outperforms the FGSM when applied to any model. The gap 
  is particularly pronounced for the adversarially trained model.
* Ensemble Adversarial Training is more robust than (standard) adversarial 
  training to transferred examples computed using any of the attacks above.

###### CONTACT
Questions and suggestions can be sent to tramer@cs.stanford.edu 
