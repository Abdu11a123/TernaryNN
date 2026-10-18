
#pragma once

#include "TernaryLibv2.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace TNN {

// ================================================================
// Bias Trainer
// ================================================================
//
// ONLY Layer::bias is trained.
//
// pos_w and neg_w are NEVER modified.
//
// ================================================================

class BiasTrainer
{
private:

    Model& model;

    // ------------------------------------------------------------
    // Hyperparameters
    // ------------------------------------------------------------

    static constexpr float K = 250.0f;

    float learning_rate;


    // ------------------------------------------------------------
    // Per-layer forward/backpropagation state
    // ------------------------------------------------------------

    struct LayerState
    {
        std::vector<float> z;
        std::vector<float> activation;
        std::vector<float> delta;

        float threshold = 0.0f;
    };

    std::vector<LayerState> states;


    // ============================================================
    // SIGMOID
    // ============================================================

    static float sigmoid(
        float x,
        float c)
    {
        const float t =
            K * (x - c);

        if (t >= 0.0f)
        {
            const float e =
                std::exp(-t);

            return 1.0f /
                   (1.0f + e);
        }
        else
        {
            const float e =
                std::exp(t);

            return e /
                   (1.0f + e);
        }
    }


    // ============================================================
    // SIGMOID DERIVATIVE
    // ============================================================

    static float sigmoid_derivative(
        float activation)
    {
        return
            K *
            activation *
            (1.0f - activation);
    }


    // ============================================================
    // READ-ONLY TERNARY WEIGHT
    // ============================================================

    static int weight_at(
        const Layer& ly,
        size_t neuron,
        size_t input_index)
    {
        const size_t word =
            input_index / 16;

        const size_t bit =
            input_index % 16;

        const uint16_t mask =
            static_cast<uint16_t>(
                1u << bit
            );

        const size_t packed_index =
            neuron *
            static_cast<size_t>(
                ly.packed_cols
            )
            + word;

        const bool positive =
            (ly.pos_w[packed_index] &
             mask) != 0;

        const bool negative =
            (ly.neg_w[packed_index] &
             mask) != 0;

        if (positive && !negative)
            return +1;

        if (negative && !positive)
            return -1;

        return 0;
    }


    // ============================================================
    // INPUT BIT
    // ============================================================

    static float input_bit(
        const std::vector<uint16_t>& input,
        size_t index)
    {
        const size_t word =
            index / 16;

        const size_t bit =
            index % 16;

        return
            (input[word] &
             static_cast<uint16_t>(
                 1u << bit
             ))
            ? 1.0f
            : 0.0f;
    }


    // ============================================================
    // TARGET BIT
    // ============================================================

    static float target_bit(
        const std::vector<uint16_t>& target,
        size_t index)
    {
        const size_t word =
            index / 16;

        const size_t bit =
            index % 16;

        return
            (target[word] &
             static_cast<uint16_t>(
                 1u << bit
             ))
            ? 1.0f
            : 0.0f;
    }


    // ============================================================
    // FORWARD
    // ============================================================
    //
    // Training forward pass:
    //
    // z = Wx + b
    //
    // c = l + 0.8(g-l)
    //
    // a = sigmoid(z,c)
    //
    // NOTE:
    //
    // Unlike normal inference, we DO NOT convert activation
    // into a hard binary bit.
    //
    // ============================================================

    void forward(
        const TrainSample& sample)
    {
        const size_t layer_count =
            model.layers.size();

        const std::vector<float>* previous =
            nullptr;

        for (size_t li = 0;
             li < layer_count;
             ++li)
        {
            Layer& ly =
                *model.layers[li];

            LayerState& state =
                states[li];


            // ----------------------------------------------------
            // Calculate z
            // ----------------------------------------------------

            for (int neuron = 0;
                 neuron < ly.neurons;
                 ++neuron)
            {
                float sum =
                    ly.bias[neuron];


                if (li == 0)
                {
                    // --------------------------------------------
                    // First layer
                    // --------------------------------------------

                    for (size_t i = 0;
                         i < static_cast<size_t>(
                             ly.inputs);
                         ++i)
                    {
                        const int w =
                            weight_at(
                                ly,
                                static_cast<size_t>(
                                    neuron),
                                i
                            );

                        if (w != 0)
                        {
                            sum +=
                                static_cast<float>(
                                    w)
                                *
                                input_bit(
                                    sample.input,
                                    i
                                );
                        }
                    }
                }
                else
                {
                    // --------------------------------------------
                    // Hidden/output layer
                    // --------------------------------------------

                    for (size_t i = 0;
                         i < static_cast<size_t>(
                             ly.inputs);
                         ++i)
                    {
                        const int w =
                            weight_at(
                                ly,
                                static_cast<size_t>(
                                    neuron),
                                i
                            );

                        if (w != 0)
                        {
                            sum +=
                                static_cast<float>(
                                    w)
                                *
                                (*previous)[i];
                        }
                    }
                }

                state.z[neuron] =
                    sum;
            }


            // ----------------------------------------------------
            // Dynamic threshold
            // ----------------------------------------------------

            float g =
                state.z[0];

            float l =
                state.z[0];

            for (int neuron = 1;
                 neuron < ly.neurons;
                 ++neuron)
            {
                const float v =
                    state.z[neuron];

                if (v > g)
                    g = v;

                if (v < l)
                    l = v;
            }


            state.threshold =
                l + 0.8f * (g - l);


            // ----------------------------------------------------
            // Sigmoid
            // ----------------------------------------------------

            for (int neuron = 0;
                 neuron < ly.neurons;
                 ++neuron)
            {
                state.activation[neuron] =
                    sigmoid(
                        state.z[neuron],
                        state.threshold
                    );
            }


            previous =
                &state.activation;
        }
    }


    // ============================================================
    // MSE FOR ONE SAMPLE
    // ============================================================

    double sample_loss(
        const TrainSample& sample)
    {
        forward(sample);

        const Layer& output =
            *model.layers.back();

        const LayerState& state =
            states.back();

        double total = 0.0;

        for (int i = 0;
             i < output.neurons;
             ++i)
        {
            const float prediction =
                state.activation[i];

            const float target =
                target_bit(
                    sample.target,
                    static_cast<size_t>(i)
                );

            const double error =
                static_cast<double>(
                    prediction - target
                );

            total +=
                error * error;
        }

        return
            total /
            static_cast<double>(
                output.neurons
            );
    }


    // ============================================================
    // BACKPROPAGATION
    // ============================================================

    void backward(
        const TrainSample& sample)
    {
        const size_t last =
            model.layers.size() - 1;


        // --------------------------------------------------------
        // Clear deltas
        // --------------------------------------------------------

        for (LayerState& state :
             states)
        {
            std::fill(
                state.delta.begin(),
                state.delta.end(),
                0.0f
            );
        }


        // ========================================================
        // OUTPUT LAYER
        // ========================================================

        Layer& output =
            *model.layers[last];

        LayerState& output_state =
            states[last];

        const float mse_scale =
            1.0f /
            static_cast<float>(
                output.neurons
            );


        for (int i = 0;
             i < output.neurons;
             ++i)
        {
            const float a =
                output_state.activation[i];

            const float y =
                target_bit(
                    sample.target,
                    static_cast<size_t>(i)
                );


            // dL/da
            const float dL_da =
                2.0f *
                (a - y) *
                mse_scale;


            // da/dz
            const float da_dz =
                sigmoid_derivative(a);


            // dL/dz
            output_state.delta[i] =
                dL_da *
                da_dz;
        }


        // ========================================================
        // HIDDEN LAYERS
        // ========================================================

        for (int li =
                 static_cast<int>(last) - 1;
             li >= 0;
             --li)
        {
            Layer& current =
                *model.layers[
                    static_cast<size_t>(li)
                ];

            Layer& next =
                *model.layers[
                    static_cast<size_t>(li + 1)
                ];

            LayerState& current_state =
                states[
                    static_cast<size_t>(li)
                ];

            LayerState& next_state =
                states[
                    static_cast<size_t>(li + 1)
                ];


            for (int i = 0;
                 i < current.neurons;
                 ++i)
            {
                float propagated =
                    0.0f;


                for (int j = 0;
                     j < next.neurons;
                     ++j)
                {
                    const int w =
                        weight_at(
                            next,
                            static_cast<size_t>(
                                j),
                            static_cast<size_t>(
                                i)
                        );


                    propagated +=
                        static_cast<float>(
                            w)
                        *
                        next_state.delta[j];
                }


                current_state.delta[i] =
                    propagated *
                    sigmoid_derivative(
                        current_state.activation[i]
                    );
            }
        }
    }


public:

    // ============================================================
    // CONSTRUCTOR
    // ============================================================

    explicit BiasTrainer(
        Model& m,
        float lr = 100.5f)
        :
        model(m),
        learning_rate(lr)
    {
        states.resize(
            model.layers.size()
        );


        for (size_t li = 0;
             li < model.layers.size();
             ++li)
        {
            Layer& ly =
                *model.layers[li];

            states[li].z.resize(
                static_cast<size_t>(
                    ly.neurons
                )
            );

            states[li].activation.resize(
                static_cast<size_t>(
                    ly.neurons
                )
            );

            states[li].delta.resize(
                static_cast<size_t>(
                    ly.neurons
                )
            );
        }
    }


    // ============================================================
    // LOSS
    // ============================================================

    double loss(
        const std::vector<TrainSample>& data)
    {
        if (data.empty())
            return 0.0;


        double total = 0.0;


        for (const TrainSample& sample :
             data)
        {
            total +=
                sample_loss(sample);
        }


        return
            total /
            static_cast<double>(
                data.size()
            );
    }


    // ============================================================
    // ONE TRAINING STEP
    // ============================================================

    size_t train_step(
        const std::vector<TrainSample>& data,
        double* before_loss = nullptr,
        double* after_loss = nullptr)
    {
        if (data.empty())
            return 0;


        const double before =
            loss(data);


        if (before_loss)
            *before_loss = before;


        // --------------------------------------------------------
        // Gradient storage
        // --------------------------------------------------------

        std::vector<
            std::vector<float>
        > gradients(
            model.layers.size()
        );


        for (size_t li = 0;
             li < model.layers.size();
             ++li)
        {
            gradients[li].assign(
                model.layers[li]->bias.size(),
                0.0f
            );
        }


        // ========================================================
        // CALCULATE GRADIENTS
        // ========================================================
        //
        // IMPORTANT:
        //
        // No bias is modified during this phase.
        //
        // No ternary weight is modified either.
        //
        // ========================================================

        for (const TrainSample& sample :
             data)
        {
            forward(sample);

            backward(sample);


            for (size_t li = 0;
                 li < model.layers.size();
                 ++li)
            {
                Layer& ly =
                    *model.layers[li];

                LayerState& state =
                    states[li];


                for (size_t i = 0;
                     i < ly.bias.size();
                     ++i)
                {
                    gradients[li][i] +=
                        state.delta[i];
                }
            }
        }


        // ========================================================
        // AVERAGE GRADIENTS
        // ========================================================

        const float inv =
            1.0f /
            static_cast<float>(
                data.size()
            );


        for (auto& layer_gradient :
             gradients)
        {
            for (float& g :
                 layer_gradient)
            {
                g *= inv;
            }
        }


        // ========================================================
        // UPDATE BIASES
        // ========================================================
        //
        // b = b - learning_rate * gradient
        //
        // ONLY biases are changed.
        //
        // ========================================================

        size_t changed = 0;


        for (size_t li = 0;
             li < model.layers.size();
             ++li)
        {
            Layer& ly =
                *model.layers[li];


            for (size_t i = 0;
                 i < ly.bias.size();
                 ++i)
            {
                const float old_bias =
                    ly.bias[i];


                const float gradient =
                    gradients[li][i];


                ly.bias[i] =
                    old_bias -
                    learning_rate *
                    gradient;


                if (ly.bias[i] != old_bias)
                    ++changed;
            }
        }


        // ========================================================
        // AFTER LOSS
        // ========================================================

        if (after_loss)
        {
            *after_loss =
                loss(data);
        }


        return changed;
    }


    // ============================================================
    // TRAIN
    // ============================================================

    void train(
        const std::vector<TrainSample>& data,
        size_t epochs)
    {
        for (size_t epoch = 0;
             epoch < epochs;
             ++epoch)
        {
            double before = 0.0;
            double after = 0.0;


            const size_t changed =
                train_step(
                    data,
                    &before,
                    &after
                );


            std::cout
                << "Epoch "
                << (epoch + 1)
                << " | MSE before = "
                << before
                << " | MSE after = "
                << after
                << " | biases changed = "
                << changed
                << '\n';
        }
    }


    // ============================================================
    // SAVE BIASES
    // ============================================================

    bool save_biases(
        const std::string& path)
    {
        std::ofstream file(
            path,
            std::ios::binary |
            std::ios::trunc
        );


        if (!file)
        {
            std::cerr
                << "Failed to open bias output file: "
                << path
                << '\n';

            return false;
        }


        for (auto& layer_ptr :
             model.layers)
        {
            Layer& ly =
                *layer_ptr;


            file.write(
                reinterpret_cast<const char*>(
                    ly.bias.data()
                ),
                static_cast<std::streamsize>(
                    ly.bias.size() *
                    sizeof(float)
                )
            );


            if (!file)
            {
                std::cerr
                    << "Error while writing biases.\n";

                return false;
            }
        }


        std::cout
            << "Trained biases saved to "
            << path
            << '\n';


        return true;
    }
};


} // namespace TNN
