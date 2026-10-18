
#include "TernaryLibv2.hpp"
#include "TernaryTrainer_sim.hpp"
#include "BiasTrainer.hpp"

#include <iostream>
#include <vector>

int main()
{
    using namespace TNN;


    // ============================================================
    // CREATE MODEL
    // ============================================================

    Model model;


    model.add_layer(
        96,
        256,
        "Layer1"
    );


    model.add_layer(
        256,
        256,
        "Layer2"
    );


    model.add_layer(
        256,
        32,
        "Output"
    );


    // ============================================================
    // LOAD EXISTING MODEL
    // ============================================================

    if (!model.load_weights(
            "pos_weights.bin",
            "neg_weights.bin",
            "biases.bin"))
    {
        std::cerr
            << "Failed to load model weights.\n";

        return 1;
    }


    std::cout
        << "Model weights loaded.\n";


    // ============================================================
    // LOAD TRAINING DATA
    // ============================================================

    std::vector<TrainSample> data;


    if (!load_training_data(
            "training_data.bin",
            data))
    {
        std::cerr
            << "Failed to load training data.\n";

        return 1;
    }


    if (data.empty())
    {
        std::cerr
            << "Training data is empty.\n";

        return 1;
    }


    // ============================================================
    // CREATE BIAS TRAINER
    // ============================================================
    //
    // Only biases are trainable.
    //
    // Ternary weights are frozen.
    //
    // ============================================================

    constexpr float LEARNING_RATE =
        0.000001f;


    BiasTrainer trainer(
        model,
        LEARNING_RATE
    );


    // ============================================================
    // TRAIN
    // ============================================================

    constexpr size_t EPOCHS =
        1000;


    std::cout
        << "\nStarting bias-only training...\n\n";


    trainer.train(
        data,
        EPOCHS
    );


    // ============================================================
    // SAVE
    // ============================================================

    if (!trainer.save_biases(
            "biases.bin"))
    {
        return 1;
    }


    std::cout
        << "\nTraining complete.\n";


    return 0;
}
