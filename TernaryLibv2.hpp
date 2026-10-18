#ifndef TERNARY_LIB_HPP
#define TERNARY_LIB_HPP

#include <iostream>
#include <vector>
#include <fstream>
#include <cstdint>
#include <string>
#include <algorithm>
#include <memory>
#include <cstring>

// ============================================================
// Portable Popcount
// ============================================================

#ifdef _MSC_VER
#include <intrin.h>
#define TNN_POPCOUNT16 __popcnt16
#else
#define TNN_POPCOUNT16 __builtin_popcount
#endif

namespace TNN {

// ============================================================
// Layer
// ============================================================

struct LayerConfig {
    int input_size;
    int neuron_count;
    std::string name;
};

class Layer {
public:

    std::vector<uint16_t> pos_w;
    std::vector<uint16_t> neg_w;
    std::vector<float> bias;

    int neurons;
    int inputs;
    int packed_cols;
    std::string name;

    Layer(int n, int i, std::string label)
        : neurons(n),
          inputs(i),
          name(std::move(label))
    {
        packed_cols = i / 16;

        pos_w.resize(n * packed_cols);
        neg_w.resize(n * packed_cols);
        bias.resize(n);
    }

    // --------------------------------------------------------
    // Load layer data
    // --------------------------------------------------------

    void load_binary(
        std::ifstream& fp,
        std::ifstream& fn,
        std::ifstream& fb)
    {
        fp.read(
            reinterpret_cast<char*>(pos_w.data()),
            pos_w.size() * sizeof(uint16_t)
        );

        fn.read(
            reinterpret_cast<char*>(neg_w.data()),
            neg_w.size() * sizeof(uint16_t)
        );

        fb.read(
            reinterpret_cast<char*>(bias.data()),
            bias.size() * sizeof(float)
        );
    }
};


// ============================================================
// Model
// ============================================================

class Model {
    friend class TernaryTrainer;
    friend class BiasTrainer;
private:

    std::vector<std::unique_ptr<Layer>> layers;

    // --------------------------------------------------------
    // PREALLOCATED INFERENCE MEMORY
    //
    // These are allocated ONCE.
    // They are reused for every prediction.
    // --------------------------------------------------------

    std::vector<uint16_t> buffer_a;
    std::vector<uint16_t> buffer_b;

    std::vector<float> z_buffer;

    // Maximum packed output size of any layer.
    size_t max_packed_size = 0;

    // Maximum neuron count of any layer.
    size_t max_neurons = 0;


public:

    Model() = default;


    // ========================================================
    // Add layer
    // ========================================================

    void add_layer(
        int inputs,
        int neurons,
        std::string name)
    {
        layers.push_back(
            std::make_unique<Layer>(
                neurons,
                inputs,
                std::move(name)
            )
        );

        // ----------------------------------------------------
        // Determine workspace requirements.
        // ----------------------------------------------------

        size_t packed_size =
            static_cast<size_t>((neurons + 15) / 16);

        max_packed_size =
            std::max(max_packed_size, packed_size);

        max_neurons =
            std::max(
                max_neurons,
                static_cast<size_t>(neurons)
            );

        // ----------------------------------------------------
        // Allocate workspace.
        //
        // This happens during MODEL CONSTRUCTION,
        // NOT during prediction.
        // ----------------------------------------------------

        buffer_a.resize(max_packed_size);
        buffer_b.resize(max_packed_size);

        z_buffer.resize(max_neurons);
    }


    // ========================================================
    // Load weights
    // ========================================================

    bool load_weights(
        const std::string& pos_path,
        const std::string& neg_path,
        const std::string& bias_path)
    {
        std::ifstream fp(
            pos_path,
            std::ios::binary
        );

        std::ifstream fn(
            neg_path,
            std::ios::binary
        );

        std::ifstream fb(
            bias_path,
            std::ios::binary
        );

        if (!fp || !fn || !fb)
            return false;

        for (auto& ly : layers)
        {
            ly->load_binary(fp, fn, fb);
        }

        return true;
    }


private:

    // ========================================================
    // Forward layer WITHOUT ANY ALLOCATION
    // ========================================================

    void forward_layer_into(
        const uint16_t* in,
        uint16_t* out,
        float* z,
        const Layer& ly)
    {
        // ----------------------------------------------------
        // Compute neuron activations
        // ----------------------------------------------------

        for (int i = 0; i < ly.neurons; ++i)
        {
            uint32_t p_cnt = 0;
            uint32_t n_cnt = 0;

            int offset =
                i * ly.packed_cols;

            for (int j = 0;
                 j < ly.packed_cols;
                 ++j)
            {
                p_cnt += TNN_POPCOUNT16(
                    in[j] &
                    ly.pos_w[offset + j]
                );

                n_cnt += TNN_POPCOUNT16(
                    in[j] &
                    ly.neg_w[offset + j]
                );
            }

            z[i] =
                static_cast<float>(p_cnt)
                -
                static_cast<float>(n_cnt)
                +
                ly.bias[i];
        }


        // ----------------------------------------------------
        // Dynamic threshold
        // ----------------------------------------------------

        float g = z[0];
        float l = z[0];

        for (int i = 1;
             i < ly.neurons;
             ++i)
        {
            float v = z[i];

            if (v > g)
                g = v;

            if (v < l)
                l = v;
        }

        float threshold =
            l + 0.8f * (g - l);


        // ----------------------------------------------------
        // Clear output buffer
        //
        // IMPORTANT:
        // Only clear the portion actually used by this layer.
        // ----------------------------------------------------

        const size_t output_words =
            static_cast<size_t>(
                (ly.neurons + 15) / 16
            );

        std::fill(
            out,
            out + output_words,
            uint16_t(0)
        );


        // ----------------------------------------------------
        // Pack binary activation
        // ----------------------------------------------------

        for (int i = 0;
             i < ly.neurons;
             ++i)
        {
            if (z[i] > threshold)
            {
                out[i / 16] |=
                    static_cast<uint16_t>(
                        1u << (i % 16)
                    );
            }
        }
    }


public:

    // ========================================================
    // ZERO-ALLOCATION PREDICTION
    //
    // The caller supplies the output buffer.
    //
    // No std::vector allocation occurs here.
    // ========================================================

    size_t predict_raw_into(
        const uint16_t* raw_ptr,
        size_t len,
        uint16_t* output)
    {
        if (layers.empty())
            return 0;

        // ----------------------------------------------------
        // Input validation
        // ----------------------------------------------------

        if (len !=
            static_cast<size_t>(
                layers[0]->packed_cols
            ))
        {
            return 0;
        }


        // ----------------------------------------------------
        // Ping-pong buffers
        //
        // Layer 1 → A
        // Layer 2 → B
        // Layer 3 → A
        // ----------------------------------------------------

        const uint16_t* current =
            raw_ptr;

        uint16_t* next =
            buffer_a.data();


        for (size_t layer_index = 0;
             layer_index < layers.size();
             ++layer_index)
        {
            Layer& ly =
                *layers[layer_index];


            // ------------------------------------------------
            // Forward pass
            // ------------------------------------------------

            forward_layer_into(
                current,
                next,
                z_buffer.data(),
                ly
            );


            // ------------------------------------------------
            // Prepare next layer
            // ------------------------------------------------

            current = next;


            if (layer_index + 1 <
                layers.size())
            {
                if (next == buffer_a.data())
                    next = buffer_b.data();
                else
                    next = buffer_a.data();
            }
        }


        // ----------------------------------------------------
        // Copy final result to caller's output buffer.
        // ----------------------------------------------------

        const size_t final_words =
            static_cast<size_t>(
                (layers.back()->neurons + 15) / 16
            );

        std::memcpy(
            output,
            current,
            final_words * sizeof(uint16_t)
        );

        return final_words;
    }


    // ========================================================
    // Convenience version using std::vector
    //
    // This STILL allocates the returned vector.
    //
    // Do NOT use this inside the performance benchmark.
    // ========================================================

    std::vector<uint16_t> predict(
        const std::vector<uint16_t>& input)
    {
        const size_t output_words =
            static_cast<size_t>(
                (layers.back()->neurons + 15) / 16
            );

        std::vector<uint16_t> output(
            output_words
        );

        predict_raw_into(
            input.data(),
            input.size(),
            output.data()
        );

        return output;
    }


    // ========================================================
    // Accessors
    // ========================================================

    size_t output_size() const
    {
        if (layers.empty())
            return 0;

        return static_cast<size_t>(
            (layers.back()->neurons + 15) / 16
        );
    }
    // ========================================================
    // Formatting Output as Binary (Base-2: 0s and 1s)
    // ========================================================

    // Returns the output as a std::string of '0's and '1's
    // ========================================================
    // Formatting Output as Binary (Math Format: 000...1)
    // ========================================================

    std::string output_to_binary_string(const std::vector<uint16_t>& output_packed) const
    {
        if (layers.empty())
            return "";

        int final_neurons = layers.back()->neurons;
        std::string bin_str;
        bin_str.reserve(final_neurons);

        // Loop BACKWARDS so Neuron 0 is printed last (on the far right)
        for (int i = final_neurons - 1; i >= 0; --i)
        {
            int word_index = i / 16;
            int bit_index = i % 16;

            // Check if the specific bit is set to 1
            if (output_packed[word_index] & (1u << bit_index))
            {
                bin_str += '1';
            }
            else
            {
                bin_str += '0';
            }
        }

        return bin_str;
    }

    // Directly prints the binary output to the console
    void print_binary_output(const std::vector<uint16_t>& output_packed) const
    {
        std::cout << output_to_binary_string(output_packed) << std::endl;
    }
};

}

#endif