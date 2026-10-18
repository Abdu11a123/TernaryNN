# Ternary Neural Network (TNN)

A C++ neural network implementation using **packed ternary weights, binary activations, bitwise operations, and evolutionary optimization**.

TNN is designed to explore compact neural network representations and efficient inference using integer bit operations instead of conventional floating-point weight multiplication.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Ternary Weight Representation](#ternary-weight-representation)
- [Inference Pipeline](#inference-pipeline)
- [Evolutionary Training](#evolutionary-training)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Build and Run](#build-and-run)
- [Model Configuration](#model-configuration)
- [Binary File Formats](#binary-file-formats)
- [Using the Inference API](#using-the-inference-api)
- [Training API](#training-api)
- [Performance Considerations](#performance-considerations)
- [Current Limitations](#current-limitations)
- [Future Improvements](#future-improvements)

---

## Overview

TNN implements a feedforward neural network in which:

- Weights take values from the ternary set \(\{-1,0,+1\}\).
- Neuron activations are represented as binary values.
- Weights are stored using two packed bit planes.
- Forward propagation uses bitwise AND and population-count operations.
- A dynamic threshold determines which neurons activate.
- An evolutionary algorithm searches for weight configurations that minimize training error.

The current example defines a network with a 96-bit input, two hidden layers containing 256 neurons each, and a 32-bit output.

The implementation separates the neural network, evolutionary trainer, and inference application.

## Key Features

### Compact weight storage

Each weight is represented using two bits distributed across positive and negative bit planes. This allows ternary weights to be stored compactly in `uint16_t` words.

### Bitwise inference

The forward pass computes positive and negative contributions using bitwise AND and population counts rather than floating-point multiplication for each weight.

### Binary activations

Each neuron produces a binary activation determined by a layer-wide dynamic threshold.

### Preallocated inference buffers

The model maintains reusable workspace buffers for intermediate activations and neuron scores. The `predict_raw_into()` API avoids dynamic memory allocation during normal inference, assuming the caller provides correctly sized buffers and the model is not modified concurrently.

### Evolutionary optimization

The trainer generates candidate models through random weight mutations, evaluates their training loss, and retains the best candidates for subsequent generations.

### Binary persistence

Weights and biases can be loaded from and saved to separate binary files.

---

## Architecture

The example inference application constructs the following network:

```text
Input
  96 bits
     |
     v
+----------------------+
| Hidden Layer 1       |
| 256 neurons          |
| Ternary weights      |
| Binary activations   |
+----------------------+
     |
     v
+----------------------+
| Hidden Layer 2       |
| 256 neurons          |
| Ternary weights      |
| Binary activations   |
+----------------------+
     |
     v
+----------------------+
| Output Layer         |
| 32 neurons           |
| Ternary weights      |
| Binary activations   |
+----------------------+
     |
     v
  32-bit output
```

The example is configured in C++ as follows:

```cpp
TNN::Model myNet;

myNet.add_layer(96, 256, "Hidden1");
myNet.add_layer(256, 256, "Hidden2");
myNet.add_layer(256, 32, "Output");
```

Each layer receives the preceding layer's binary output, except the first layer, which receives the external input.

### Important distinction

Although the network uses ternary weights and binary activations, its neuron scores and biases are floating-point values. This is not a purely integer network.

---

## How It Works

### 1. Ternary weights

Every connection has one of three possible weight values:

\[
w_{ij}\in\{-1,0,+1\}
\]

- \(+1\): positive contribution.
- \(-1\): negative contribution.
- \(0\): no contribution.

### 2. Binary input and activations

Each input and intermediate activation is represented as a bit:

\[
x_j\in\{0,1\}
\]

The neuron score is computed as:

\[
z_i=\sum_j w_{ij}x_j+b_i
\]

Because the input and weights are binary-valued or ternary-valued, the weighted sum can be evaluated through bitwise operations.

### 3. Dynamic threshold

For each layer, the implementation finds the maximum and minimum neuron scores:

\[
g=\max_i z_i,\qquad l=\min_i z_i
\]

It then calculates:

\[
\theta=l+0.8(g-l)
\]

The activation rule is:

\[
a_i=
\begin{cases}
1 & z_i>\theta\\
0 & z_i\leq\theta
\end{cases}
\]

The resulting activations are packed into `uint16_t` words, with each word storing up to 16 binary activations.

**Note:** This threshold is calculated independently for every layer and every inference. It is not a fixed threshold of zero.

---

## Ternary Weight Representation

The implementation stores weights using two bit planes:

- `pos_w`: indicates positive weights.
- `neg_w`: indicates negative weights.

For each connection, the logical representation is:

| `pos_w` bit | `neg_w` bit | Logical weight |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 0 | +1 |
| 0 | 1 | -1 |
| 1 | 1 | Invalid or noncanonical |

The current inference calculation subtracts negative-plane matches from positive-plane matches. Therefore, if both bits are set, the two contributions cancel. The trainer's mutation functions, however, maintain canonical ternary states and never deliberately create this combination.

### Memory usage

Two `uint16_t` bit planes encode 16 weights per plane word, requiring 32 bits in total for 16 weights.

The effective weight-storage cost is:

\[
\frac{32}{16}=2\text{ bits per weight}
\]

For a layer with \(N\) neurons and \(I\) inputs, the nominal weight-storage size is:

\[
M_{\text{weights}}=
2N\left\lceil\frac{I}{16}\right\rceil
\text{ bytes}
\]

This expression assumes the packed input width is rounded up to the next 16-bit word.

**Implementation note:** The current `Layer` constructor calculates `packed_cols` using `i / 16`, which rounds down. Consequently, input sizes that are not divisible by 16 are not represented correctly. Use input widths divisible by 16 until this is fixed.

---

## Inference Pipeline

The inference implementation resides in `TernaryLibv2.hpp`.

### Forward pass

For each neuron, the implementation:

1. ANDs the packed input with the positive weight plane.
2. Counts the set bits.
3. ANDs the packed input with the negative weight plane.
4. Counts those set bits.
5. Subtracts the negative contribution from the positive contribution.
6. Adds the neuron's bias.
7. Applies the layer's dynamic threshold.
8. Packs the resulting binary activations.

The core score calculation is equivalent to:

```cpp
z[i] =
    static_cast<float>(p_cnt)
    - static_cast<float>(n_cnt)
    + ly.bias[i];
```

### Public inference methods

#### `predict_raw_into()`

```cpp
size_t predict_raw_into(
    const uint16_t* input,
    size_t input_words,
    uint16_t* output
);
```

This method writes predictions into a caller-provided buffer and returns the number of output words.

For the example network:

- Input: 6 `uint16_t` words.
- Output: 2 `uint16_t` words.

It is the preferred API for repeated inference when avoiding output-vector allocations matters.

#### `predict()`

```cpp
std::vector<uint16_t> predict(
    const std::vector<uint16_t>& input
);
```

This convenience API returns the packed output as a vector. It allocates storage for the output vector.

#### `output_to_binary_string()`

```cpp
std::string output_to_binary_string(
    const std::vector<uint16_t>& output
) const;
```

Converts packed activations into a string containing `0` and `1` characters.

The string is printed with the highest-indexed neuron first and neuron zero at the rightmost position.

#### `print_binary_output()`

```cpp
void print_binary_output(
    const std::vector<uint16_t>& output
) const;
```

Prints the formatted binary output.

---

## Evolutionary Training

The trainer is implemented in the training header and exposes the `TNN::TernaryTrainer` class.

It uses a mutation-based evolutionary search rather than gradient descent.

### Initial population

The trainer clones the current model 128 times and applies mutation rates indexed from 0 through 127:

\[
r_i=\frac{i}{128}
\]

The number of selected weights is:

\[
m_i=\operatorname{round}(r_iW)
\]

where \(W\) is the total number of trainable weights.

The candidates are evaluated, sorted by loss, and the best 16 are retained.

### Subsequent generations

Each retained parent produces 16 children:

- Child 0: no mutations.
- Child 1: approximately \(1/128\) of all weights selected.
- ...
- Child 15: approximately \(15/128\) of all weights selected.

With 16 parents, this produces:

\[
16\times16=256
\]

candidate models per generation.

The 256 candidates are evaluated, and the best 16 become the parents of the next generation.

### Mutation mechanics

For every mutation:

1. A weight index is selected.
2. The current ternary value is read.
3. The weight is changed to one of its other two legal states.
4. The updated value is written into the packed bit planes.

A partial Fisher–Yates shuffle selects distinct indices within each mutation operation, preventing a single mutation operation from selecting the same weight twice.

### Fitness function

The trainer uses total bitwise Hamming loss across the dataset:

\[
L=\sum_{s=1}^{S}\sum_{k=1}^{K}
\operatorname{popcount}
\left(\hat y_{s,k}\oplus y_{s,k}\right)
\]

where:

- \(S\) is the number of training samples.
- \(K\) is the number of packed output words.
- \(\hat y\) is the predicted output.
- \(y\) is the target output.
- \(\oplus\) is bitwise XOR.

A loss of zero means every compared output bit matches the corresponding target bit.

### Training API

```cpp
TNN::TernaryTrainer trainer(myNet, 12345);

trainer.train(training_data, 50);

trainer.save_weights(
    "pos_weights.bin",
    "neg_weights.bin",
    "biases.bin"
);
```

The integer seed makes the pseudorandom mutation sequence reproducible for a given implementation and execution environment, assuming the same initial model, data, and random-distribution behavior.

The `cycles` argument specifies the maximum number of generations after the initial population search.

**Important:** The current trainer mutates weights, not biases. Biases remain fixed during evolutionary training. This can substantially constrain the search space.

---

## Project Structure

A recommended organization for the supplied code is:

```text
TNN/
├── TernaryLibv2.hpp
├── TernaryTrainer.hpp
├── main.cpp
├── pos_weights.bin
├── neg_weights.bin
├── biases.bin
├── inputs.bin
├── training_data.bin
└── README.md
```

The training header should include `TernaryLibv2.hpp`, and the inference application should include the same model header.

The exact filenames can differ from this suggested layout.

---

## Requirements

- A C++ compiler with C++17 support recommended.
- A standard library implementation supporting `std::vector`, `std::unique_ptr`, random-number facilities, and file streams.
- Binary model files compatible with the network architecture.
- Input data encoded in the expected binary format.

The implementation includes compiler-specific population-count macros for MSVC and GCC/Clang-style compilers. Other toolchains may require an appropriate population-count implementation.

---

## Build and Run

### 1. Compile the inference application

For GCC or Clang, assuming the files are in the same directory:

```bash
g++ -std=c++17 -O3 -DNDEBUG main.cpp -o tnn
```

On Windows using MinGW:

```bash
g++ -std=c++17 -O3 -DNDEBUG main.cpp -o tnn.exe
```

### 2. Prepare model files

Place the following files in the working directory:

```text
pos_weights.bin
neg_weights.bin
biases.bin
```

These files must correspond to the architecture configured in `main.cpp`.

### 3. Prepare input data

The inference application expects `inputs.bin` to contain consecutive records, each consisting of six `uint16_t` words.

Each record represents a 96-bit input.

### 4. Execute

On Linux:

```bash
./tnn
```

On Windows:

```bash
tnn.exe
```

For every complete input record, the application prints the binary prediction followed by its packed 16-bit output words.

---

## Model Configuration

The model is constructed by adding layers in sequence:

```cpp
TNN::Model model;

model.add_layer(96, 256, "Hidden1");
model.add_layer(256, 256, "Hidden2");
model.add_layer(256, 32, "Output");
```

The arguments are:

```cpp
add_layer(input_count, neuron_count, layer_name);
```

The input count for every layer must match the preceding layer's neuron count.

For the supplied packed representation, every layer's input width should be divisible by 16.

### Loading weights

```cpp
bool loaded = model.load_weights(
    "pos_weights.bin",
    "neg_weights.bin",
    "biases.bin"
);
```

The three files are read sequentially, layer by layer, in the order in which the layers were added.

The files must use the same architecture, layer ordering, data types, and binary layout as the current implementation.

---

## Binary File Formats

### Model weights

The model uses three separate binary files.

| File | Stored values | Type |
|---|---|---|
| `pos_weights.bin` | Positive weight bit planes | `uint16_t` |
| `neg_weights.bin` | Negative weight bit planes | `uint16_t` |
| `biases.bin` | Neuron biases | `float` |

For each layer, the expected order within each file is:

1. All packed data for the first layer.
2. All packed data for the second layer.
3. All packed data for the third layer.
4. Continue for any additional layers.

Each layer stores packed weights in neuron-major order, with each neuron's packed input words stored consecutively.

Biases are stored in neuron order for each layer.

The format currently uses native binary representations. It does not define a portable endianness, version header, or architecture metadata.

### Input file

`inputs.bin` consists of repeated fixed-size input records.

Each input record contains:

| Field | Number of words | Data type |
|---|---:|---|
| Input | 6 | `uint16_t` |
| Total | 6 | 12 bytes |

The file should contain an integer number of complete records.

### Training dataset

The training data loader expects `training_data.bin` to contain repeated records of eight 16-bit values:

| Field | Number of words | Data type |
|---|---:|---|
| Input | 6 | `int16_t` |
| Target | 2 | `int16_t` |
| Total | 8 | 16 bytes |

The loader converts the signed values to `uint16_t`, preserving their 16-bit patterns.

This is important because the data represents packed bit patterns, not conventional signed numerical features.

The training targets are compared against predicted output words using XOR and population count.

---

## Using the Inference API

A minimal inference example:

```cpp
#include "TernaryLibv2.hpp"

#include <cstdint>
#include <iostream>
#include <vector>

int main()
{
    TNN::Model model;

    model.add_layer(96, 256, "Hidden1");
    model.add_layer(256, 256, "Hidden2");
    model.add_layer(256, 32, "Output");

    if (!model.load_weights(
            "pos_weights.bin",
            "neg_weights.bin",
            "biases.bin"))
    {
        std::cerr << "Failed to load model.\n";
        return 1;
    }

    std::vector<uint16_t> input(6, 0);

    std::vector<uint16_t> output(
        model.output_size()
    );

    const size_t words = model.predict_raw_into(
        input.data(),
        input.size(),
        output.data()
    );

    if (words == 0)
    {
        std::cerr << "Prediction failed.\n";
        return 1;
    }

    model.print_binary_output(output);

    return 0;
}
```

The example initializes the input to zero. Replace it with the packed input values required by your application.

---

## Training API

The training header provides two main interfaces.

### Loading training data

```cpp
std::vector<TNN::TrainSample> training_data;

if (!TNN::load_training_data(
        "training_data.bin",
        training_data))
{
    return 1;
}
```

Each `TrainSample` contains:

```cpp
struct TrainSample
{
    std::vector<uint16_t> input;
    std::vector<uint16_t> target;
};
```

### Running training

```cpp
TNN::Model model;

model.add_layer(96, 256, "Hidden1");
model.add_layer(256, 256, "Hidden2");
model.add_layer(256, 32, "Output");

// Initialize or load the starting model before training.

TNN::TernaryTrainer trainer(model, 12345);

trainer.train(training_data, 50);

if (!trainer.save_weights(
        "pos_weights.bin",
        "neg_weights.bin",
        "biases.bin"))
{
    return 1;
}
```

Training starts from the model's existing parameters. It does not initialize weights through a specialized neural-network initialization scheme.

For useful results, the starting model and dataset must have compatible dimensions and valid packed weight representations.

---

## Performance Considerations

The implementation is designed to reduce the cost of inference and investigate compact neural computation.

### Potential advantages

- Two-bit logical weight representation.
- Bitwise computation of positive and negative contributions.
- Population-count operations for parallel bit processing.
- Packed binary intermediate activations.
- Reusable inference workspaces.
- Caller-provided output storage for allocation-conscious inference.

### Important trade-offs

The network still computes floating-point neuron scores and uses floating-point biases and thresholds. Its performance will depend on the processor's population-count throughput, memory bandwidth, compiler optimizations, and layer dimensions.

Evolutionary training can also be expensive. Each candidate evaluates the entire dataset, and cloning a model copies all its layers and parameters.

For the example architecture, the trainable connection count is:

\[
96(256)+256(256)+256(32)=98{,}304
\]

A generation evaluates 256 candidates across the complete training dataset, so training cost can grow quickly with dataset size.

Benchmark inference and training separately before making performance claims.

---

## Current Limitations

The following issues should be considered before treating this implementation as production-ready.

### 1. Input widths must be divisible by 16

The current layer constructor uses:

```cpp
packed_cols = i / 16;
```

This truncates rather than rounds up. A width such as 20 inputs would allocate space for only one packed word per neuron.

Use:

```cpp
packed_cols = (i + 15) / 16;
```

For non-multiple-of-16 widths, also ensure unused padding bits are masked or guaranteed to remain zero.

### 2. Model loading does not fully validate file sizes

The current loader checks that the files open, but it does not verify that all expected bytes were successfully read or that unexpected trailing data is absent.

Malformed or truncated model files should be rejected explicitly.

### 3. Model copying assumes matching architectures

`copy_model_into_current()` checks only the number of layers. It assumes the corresponding layers have compatible dimensions and parameter storage.

The model should validate each layer's dimensions before copying.

### 4. Training does not optimize biases

The evolutionary trainer mutates ternary weights but leaves biases unchanged. The threshold is recomputed during inference, but this does not make the biases trainable.

### 5. The dynamic threshold has edge cases

The threshold is based on the minimum and maximum score. When all neuron scores are equal, no neuron satisfies the strict `z[i] > threshold` activation condition, so the layer outputs all zeros.

This behavior may be undesirable for some tasks.

### 6. Hamming loss treats every output bit equally

The fitness function minimizes total bit mismatches. It does not directly optimize application-level accuracy, class-level cross-entropy, or other task-specific metrics.

### 7. No validation split or checkpointing

The current trainer evaluates candidates on the supplied training dataset only. It does not automatically track validation performance, save intermediate checkpoints, or implement early stopping based on generalization.

### 8. Packed-file portability

The binary format uses native representations of `uint16_t`, `int16_t`, and `float`. Files are not guaranteed to be portable across architectures with different byte orders or floating-point representations.

### 9. Concurrency

The inference workspace buffers are stored inside the model and reused for each prediction. Concurrent calls on the same model instance can race unless access is synchronized or each thread uses its own model/workspace.

---

## Future Improvements

Potential next steps include:

- Correct packed-width handling and validation.
- Add a versioned binary model format with explicit metadata.
- Add robust error handling for truncated or incompatible model files.
- Evolve biases alongside weights.
- Preserve parent elites explicitly between generations.
- Add mutation strategies with adaptive rates.
- Track validation loss and generalization performance.
- Add checkpointing and model evaluation reports.
- Benchmark against equivalent conventional neural networks.
- Explore SIMD acceleration and architecture-specific population-count instructions.
- Add unit tests for ternary encoding, packing, inference, serialization, and mutation.
- Compare evolutionary search with gradient-based or hybrid optimization methods.

---

## License

No license is specified in the supplied source code. Add a license file before distributing or reusing the project publicly.

---

## Summary

TNN is an experimental C++ feedforward neural network built around packed ternary weights, binary activations, population-count inference, and evolutionary weight optimization.

Its primary areas of interest are compact parameter representation, bitwise computation, and derivative-free optimization. Correctness testing, binary-format validation, and systematic benchmarking are the next steps toward evaluating its practical performance and reliability.
