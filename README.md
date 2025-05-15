# Train to Defend: First Defense Against Cryptanalytic Parameter Extraction in Neural Networks 

This repository presents the first defense for securing ReLU-based deep neural networks against cryptanalytic parameter recovery attacks.
Our core idea is to break the neuron uniqueness these attacks rely on by introducing a novel, extraction-aware training strategy.
The defense adds zero area-delay overhead during inference and causes only a marginal change in model accuracy.

# Explanation of this Codebase

This codebase implements a defense mechanism against the end-to-end cryptanatic attack proposed by Foerster et al., whose original code can be found at https://github.com/hannafoe/cryptanalytical-extraction. At the core of our defense is a custom Keras loss function, `CombinedLoss`, which improves neural network resilience by regularizing neuron similarity. Alongside the standard mean squared error loss, it adds a similarity term that penalizes large differences between randomly selected pairs of neurons within each layer. This discourages neuron uniqueness—an essential factor exploited in the original attack thereby making parameter recovery significantly more difficult.

# Reproduce Attacks
To reproduce the attack results reported in our manuscript, please execute the following commands:
Attack on the baseline MNIST model (8 hidden layers, each with 16 neurons):
```
python -u -m neuronWiggle --model models/mnist784_16x8_1_Seed42.keras --layerID 1 --seed 20 --dataset 'mnist' --quantized 2 2>&1 > .\results\mnist784_16x8_1_Seed42.txt              
```
Attack on the protected MNIST model (8 hidden layers, each with 16 neurons):
```
python -u -m neuronWiggle --model models/Secure_mnist784_16x8_1_Seed42.keras --layerID 1 --seed 20 --dataset 'mnist' --quantized 2 2>&1 > .results\Secure_mnist784_16x8_1_Seed42.txt
```
The `--seed` option lets you run the extraction using different random seeds, which can help test how results vary. The `--quantized` option controls the precision used for sign extraction: use 1 for float16, 2 for float32 (which is the default), and 0 for float64. When float64 is selected, an additional step is performed to improve precision during signature recovery. This step is skipped when using float16 or float32. The `--signRecoveryMethod` option lets you choose between two methods: `neuronWiggle` (used by default) or `carlini`. If you only want to run the sign recovery step and skip the full signature recovery, you can set `--onlySign` to True.

# Create a new model

A new model can be created by running the script `models\make_new_models.py` using the command:
```
python .\models\make_new_models.py
```
Before executing this script, ensure that the desired function is uncommented in the make_new_models.py file. Comments within the file indicate where to uncomment for creating new MNIST models or random models. Within these functions, you can specify parameters such as the number of hidden layers, the number of neurons per layer, and the training seed. For functions that generate protected models, you should also specify the impact factor (λ_similarity) that controls the strength of the security term added to the loss function.

# Dependencies

The code execution relies on the Python modules denoted below. The experiments were run on Python 3.10.11. In case the code does not run, here are the versions used. For more details please refer to the requirements.txt file.

```
pip install tensorflow==2.15.0
pip install numpy==1.25.2
pip install pandas==2.1.2
pip install jax==0.4.30
pip install optax==0.2.3
pip install scipy==1.9.3
```
