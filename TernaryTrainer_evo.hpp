#pragma once

#include "TernaryLibv2.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <random>
#include <cmath>
#include <numeric>

namespace TNN {

// ============================================================================
// Training Sample
// ============================================================================

struct TrainSample
{
    std::vector<uint16_t> input;
    std::vector<uint16_t> target;
};


// ============================================================================
// Evolutionary Ternary Trainer
// ============================================================================
//
// Algorithm:
//
// 1. Start from the current model.
//
// 2. Generate 128 candidates:
//
//       mutation rate = 0/128
//                      1/128
//                      2/128
//                      ...
//                      127/128
//
// 3. Evaluate all 128.
//
// 4. Keep the best 16.
//
// 5. Each of those 16 parents produces 16 children:
//
//       mutation rate = 0/128
//                      1/128
//                      ...
//                      15/128
//
//    Total = 16 * 16 = 256 candidates.
//
// 6. Evaluate all 256.
//
// 7. Keep the best 16.
//
// 8. Repeat.
//
// Mutation:
//
//     Randomly select N DISTINCT weights.
//
//     Each selected ternary weight is changed to one
//     of its OTHER TWO ternary states.
//
// ============================================================================

class TernaryTrainer
{
private:

    Model& model;

    // Reused prediction buffer for loss evaluation.
    std::vector<uint16_t> prediction;

    // Random generator.
    std::mt19937_64 rng;

    // ------------------------------------------------------------------------
    // Network constants
    // ------------------------------------------------------------------------

    static constexpr size_t INITIAL_POPULATION = 128;
    static constexpr size_t ELITE_COUNT        = 16;
    static constexpr size_t CHILDREN_PER_PARENT = 16;

    // Mutation rates are i / 128.
    static constexpr size_t MUTATION_DIVISOR = 128;


    // ========================================================================
    // Candidate
    // ========================================================================

    struct Candidate
    {
        std::unique_ptr<Model> model;
        double loss = std::numeric_limits<double>::infinity();
    };


    // ========================================================================
    // Calculate total number of trainable weights
    // ========================================================================

    static size_t count_weights(const Model& m)
    {
        size_t total = 0;

        for (const auto& layer_ptr : m.layers)
        {
            const Layer& ly = *layer_ptr;

            total +=
                static_cast<size_t>(ly.neurons) *
                static_cast<size_t>(ly.inputs);
        }

        return total;
    }


    // ========================================================================
    // Clone model
    // ========================================================================
    //
    // Creates a complete independent copy of the current model.
    //
    // ========================================================================

    static std::unique_ptr<Model> clone_model(const Model& source)
    {
        auto result =
            std::make_unique<Model>();

        for (const auto& layer_ptr : source.layers)
        {
            const Layer& src = *layer_ptr;

            result->add_layer(
                src.inputs,
                src.neurons,
                src.name
            );

            Layer& dst =
                *result->layers.back();

            dst.pos_w = src.pos_w;
            dst.neg_w = src.neg_w;
            dst.bias  = src.bias;
        }

        return result;
    }


    // ========================================================================
    // Convert flat weight index into packed location
    // ========================================================================
    //
    // Global flat index:
    //
    // [Layer 0 weights]
    // [Layer 1 weights]
    // [Layer 2 weights]
    //
    // ========================================================================

    static void locate_weight(
        Model& m,
        size_t global_index,
        Layer*& layer,
        size_t& local_index)
    {
        size_t offset = 0;

        for (auto& layer_ptr : m.layers)
        {
            Layer& ly = *layer_ptr;

            const size_t layer_weights =
                static_cast<size_t>(ly.neurons) *
                static_cast<size_t>(ly.inputs);

            if (global_index < offset + layer_weights)
            {
                layer = &ly;
                local_index =
                    global_index - offset;

                return;
            }

            offset += layer_weights;
        }

        layer = nullptr;
        local_index = 0;
    }


    // ========================================================================
    // Get ternary weight
    // ========================================================================

    static int get_weight(
        const Layer& ly,
        size_t flat_index)
    {
        const size_t neuron =
            flat_index /
            static_cast<size_t>(ly.inputs);

        const size_t input_index =
            flat_index %
            static_cast<size_t>(ly.inputs);

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
            static_cast<size_t>(ly.packed_cols)
            + word;

        const bool positive =
            (ly.pos_w[packed_index] & mask) != 0;

        const bool negative =
            (ly.neg_w[packed_index] & mask) != 0;

        if (positive && !negative)
            return +1;

        if (negative && !positive)
            return -1;

        return 0;
    }


    // ========================================================================
    // Set ternary weight
    // ========================================================================

    static void set_weight(
        Layer& ly,
        size_t flat_index,
        int value)
    {
        const size_t neuron =
            flat_index /
            static_cast<size_t>(ly.inputs);

        const size_t input_index =
            flat_index %
            static_cast<size_t>(ly.inputs);

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
            static_cast<size_t>(ly.packed_cols)
            + word;

        // Remove old state from both planes.
        ly.pos_w[packed_index] &=
            static_cast<uint16_t>(~mask);

        ly.neg_w[packed_index] &=
            static_cast<uint16_t>(~mask);

        // Write new state.
        if (value == +1)
        {
            ly.pos_w[packed_index] |= mask;
        }
        else if (value == -1)
        {
            ly.neg_w[packed_index] |= mask;
        }
    }


    // ========================================================================
    // Mutate one model
    // ========================================================================
    //
    // Exactly N DISTINCT weights are selected.
    //
    // Each selected weight changes to one of the OTHER TWO ternary states.
    //
    // ========================================================================

    void mutate(
        Model& candidate,
        size_t mutation_count)
    {
        const size_t total_weights =
            count_weights(candidate);

        if (mutation_count == 0)
            return;

        if (mutation_count > total_weights)
            mutation_count = total_weights;

        // --------------------------------------------------------------------
        // We use a partial Fisher-Yates shuffle.
        //
        // indices[i] contains the remaining candidate weight indices.
        //
        // Only the first mutation_count positions are randomized.
        //
        // This guarantees DISTINCT weight indices.
        // --------------------------------------------------------------------

        std::vector<size_t> indices(total_weights);

        std::iota(
            indices.begin(),
            indices.end(),
            size_t(0)
        );

        for (size_t i = 0;
             i < mutation_count;
             ++i)
        {
            std::uniform_int_distribution<size_t> dist(
                i,
                total_weights - 1
            );

            const size_t selected =
                dist(rng);

            std::swap(
                indices[i],
                indices[selected]
            );
        }

        // --------------------------------------------------------------------
        // Mutate selected weights.
        // --------------------------------------------------------------------

        for (size_t i = 0;
             i < mutation_count;
             ++i)
        {
            Layer* layer = nullptr;
            size_t local_index = 0;

            locate_weight(
                candidate,
                indices[i],
                layer,
                local_index
            );

            if (!layer)
                continue;

            const int current =
                get_weight(
                    *layer,
                    local_index
                );

            int new_value = current;

            // ----------------------------------------------------------------
            // Choose randomly from the TWO OTHER ternary states.
            //
            // -1 -> 0 or +1
            //  0 -> -1 or +1
            // +1 -> -1 or 0
            // ----------------------------------------------------------------

            if (current == -1)
            {
                std::uniform_int_distribution<int> d(0, 1);

                new_value =
                    (d(rng) == 0)
                    ? 0
                    : +1;
            }
            else if (current == 0)
            {
                std::uniform_int_distribution<int> d(0, 1);

                new_value =
                    (d(rng) == 0)
                    ? -1
                    : +1;
            }
            else
            {
                std::uniform_int_distribution<int> d(0, 1);

                new_value =
                    (d(rng) == 0)
                    ? -1
                    : 0;
            }

            set_weight(
                *layer,
                local_index,
                new_value
            );
        }
    }


    // ========================================================================
    // Calculate mutation count
    // ========================================================================

    static size_t mutation_count_for_rate(
        size_t rate_index,
        size_t total_weights)
    {
        const double rate =
            static_cast<double>(rate_index) /
            static_cast<double>(MUTATION_DIVISOR);

        return static_cast<size_t>(
            std::llround(
                rate *
                static_cast<double>(total_weights)
            )
        );
    }


    // ========================================================================
    // Loss
    // ========================================================================

    double loss(
        Model& candidate,
        const std::vector<TrainSample>& data)
    {
        double total = 0.0;

        for (const auto& sample : data)
        {
            const size_t words =
                candidate.predict_raw_into(
                    sample.input.data(),
                    sample.input.size(),
                    prediction.data()
                );

            if (words != sample.target.size())
            {
                return
                    std::numeric_limits<double>::infinity();
            }

            // ---------------------------------------------------------------
            // Hamming loss over all 32 output bits.
            // ---------------------------------------------------------------

            for (size_t w = 0;
                 w < words;
                 ++w)
            {
                const uint16_t diff =
                    static_cast<uint16_t>(
                        prediction[w] ^
                        sample.target[w]
                    );

                total += static_cast<double>(
                    TNN_POPCOUNT16(diff)
                );
            }
        }

        return total;
    }


    // ========================================================================
    // Evaluate population
    // ========================================================================

    void evaluate_population(
        std::vector<Candidate>& population,
        const std::vector<TrainSample>& data)
    {
        for (Candidate& candidate : population)
        {
            candidate.loss =
                loss(
                    *candidate.model,
                    data
                );
        }
    }


    // ========================================================================
    // Keep best N
    // ========================================================================

    void sort_and_keep_best(
        std::vector<Candidate>& population,
        size_t keep)
    {
        std::sort(
            population.begin(),
            population.end(),
            [](const Candidate& a,
               const Candidate& b)
            {
                return a.loss < b.loss;
            }
        );

        if (population.size() > keep)
        {
            population.resize(keep);
        }
    }


    // ========================================================================
    // Create initial population
    // ========================================================================

    std::vector<Candidate> create_initial_population()
    {
        std::vector<Candidate> population;

        population.reserve(
            INITIAL_POPULATION
        );

        const size_t total_weights =
            count_weights(model);

        // --------------------------------------------------------------------
        // 128 mutation rates:
        //
        // 0/128
        // 1/128
        // ...
        // 127/128
        // --------------------------------------------------------------------

        for (size_t i = 0;
             i < INITIAL_POPULATION;
             ++i)
        {
            Candidate candidate;

            candidate.model =
                clone_model(model);

            const size_t mutations =
                mutation_count_for_rate(
                    i,
                    total_weights
                );

            mutate(
                *candidate.model,
                mutations
            );

            population.push_back(
                std::move(candidate)
            );
        }

        return population;
    }


    // ========================================================================
    // Create next generation
    // ========================================================================
    //
    // Each of the 16 parents creates 16 children.
    //
    // Mutation rates:
    //
    // 0/128 ... 15/128
    //
    // The 0% child is therefore an exact copy of its parent.
    //
    // ========================================================================

    std::vector<Candidate> create_next_generation(
        const std::vector<Candidate>& parents)
    {
        std::vector<Candidate> population;

        population.reserve(
            ELITE_COUNT *
            CHILDREN_PER_PARENT
        );

        const size_t total_weights =
            count_weights(
                *parents[0].model
            );

        for (size_t parent_index = 0;
             parent_index < parents.size();
             ++parent_index)
        {
            const Model& parent =
                *parents[parent_index].model;

            for (size_t child_index = 0;
                 child_index < CHILDREN_PER_PARENT;
                 ++child_index)
            {
                Candidate child;

                child.model =
                    clone_model(parent);

                const size_t mutations =
                    mutation_count_for_rate(
                        child_index,
                        total_weights
                    );

                mutate(
                    *child.model,
                    mutations
                );

                population.push_back(
                    std::move(child)
                );
            }
        }

        return population;
    }


    // ========================================================================
    // Replace current model with winner
    // ========================================================================

    void copy_model_into_current(
        const Model& source)
    {
        // ---------------------------------------------------------------
        // Architecture is assumed identical.
        // ---------------------------------------------------------------

        if (model.layers.size() != source.layers.size())
            return;

        for (size_t i = 0;
             i < model.layers.size();
             ++i)
        {
            Layer& dst =
                *model.layers[i];

            const Layer& src =
                *source.layers[i];

            dst.pos_w = src.pos_w;
            dst.neg_w = src.neg_w;
            dst.bias  = src.bias;
        }
    }


public:

    // ========================================================================
    // Constructor
    // ========================================================================

    explicit TernaryTrainer(
        Model& m,
        uint64_t seed =
            std::random_device{}())
        : model(m),
          rng(seed)
    {
        prediction.resize(
            model.output_size()
        );
    }


    // ========================================================================
    // SAVE WEIGHTS
    // ========================================================================

    bool save_weights(
        const std::string& pos_path,
        const std::string& neg_path,
        const std::string& bias_path)
    {
        std::ofstream fp(
            pos_path,
            std::ios::binary |
            std::ios::trunc
        );

        std::ofstream fn(
            neg_path,
            std::ios::binary |
            std::ios::trunc
        );

        std::ofstream fb(
            bias_path,
            std::ios::binary |
            std::ios::trunc
        );

        if (!fp || !fn || !fb)
        {
            std::cerr
                << "Failed to open weight files for saving.\n";

            return false;
        }

        for (auto& layer_ptr : model.layers)
        {
            Layer& ly = *layer_ptr;

            fp.write(
                reinterpret_cast<const char*>(
                    ly.pos_w.data()
                ),
                static_cast<std::streamsize>(
                    ly.pos_w.size() *
                    sizeof(uint16_t)
                )
            );

            fn.write(
                reinterpret_cast<const char*>(
                    ly.neg_w.data()
                ),
                static_cast<std::streamsize>(
                    ly.neg_w.size() *
                    sizeof(uint16_t)
                )
            );

            fb.write(
                reinterpret_cast<const char*>(
                    ly.bias.data()
                ),
                static_cast<std::streamsize>(
                    ly.bias.size() *
                    sizeof(float)
                )
            );

            if (!fp || !fn || !fb)
            {
                std::cerr
                    << "Error while writing weights.\n";

                return false;
            }
        }

        return true;
    }


    // ========================================================================
    // RUN EVOLUTIONARY TRAINING
    // ========================================================================
    //
    // cycles = number of generations AFTER the initial 128-model search.
    //
    // Example:
    //
    // train(data, 50)
    //
    // means:
    //
    // Initial 128 candidates
    //       ↓
    // select 16
    //       ↓
    // Generation 1: 256
    //       ↓
    // select 16
    //       ↓
    // Generation 2: 256
    //       ↓
    // ...
    //
    // ========================================================================

    void train(
        const std::vector<TrainSample>& data,
        size_t cycles)
    {
        if (data.empty())
        {
            std::cerr
                << "Training dataset is empty.\n";

            return;
        }

        const size_t total_weights =
            count_weights(model);

        std::cout
            << "\n========================================\n"
            << " TERNARY EVOLUTIONARY TRAINER\n"
            << "========================================\n"
            << "Trainable weights : "
            << total_weights
            << '\n'
            << "Initial models    : "
            << INITIAL_POPULATION
            << '\n'
            << "Elite models      : "
            << ELITE_COUNT
            << '\n'
            << "Children/parent   : "
            << CHILDREN_PER_PARENT
            << '\n'
            << "Next population   : "
            << ELITE_COUNT * CHILDREN_PER_PARENT
            << '\n'
            << "Mutation divisor  : "
            << MUTATION_DIVISOR
            << "\n\n";


        // ====================================================================
        // INITIAL SEARCH
        // ====================================================================

        std::cout
            << "Generating initial 128 models...\n";

        std::vector<Candidate> population =
            create_initial_population();

        std::cout
            << "Evaluating initial population...\n";

        evaluate_population(
            population,
            data
        );

        sort_and_keep_best(
            population,
            ELITE_COUNT
        );

        std::cout
            << "\nInitial search complete.\n"
            << "Best loss = "
            << population[0].loss
            << "\n";


        // ====================================================================
        // EVOLUTIONARY CYCLES
        // ====================================================================

        for (size_t cycle = 0;
             cycle < cycles;
             ++cycle)
        {
            std::cout
                << "\n----------------------------------------\n"
                << "Generation "
                << (cycle + 1)
                << '\n'
                << "----------------------------------------\n";

            const double parent_best =
                population[0].loss;


            // ---------------------------------------------------------------
            // Generate 256 children.
            // ---------------------------------------------------------------

            std::vector<Candidate> children =
                create_next_generation(
                    population
                );

            std::cout
                << "Generated "
                << children.size()
                << " candidate models.\n";

            // ---------------------------------------------------------------
            // Evaluate all children.
            // ---------------------------------------------------------------

            evaluate_population(
                children,
                data
            );

            // ---------------------------------------------------------------
            // Keep best 16.
            // ---------------------------------------------------------------

            sort_and_keep_best(
                children,
                ELITE_COUNT
            );

            population =
                std::move(children);


            const double best =
                population[0].loss;


            std::cout
                << "Best loss      = "
                << best
                << '\n'
                << "Previous best  = "
                << parent_best
                << '\n'
                << "Improvement     = "
                << (parent_best - best)
                << '\n';


            // ---------------------------------------------------------------
            // Show the 16 elite losses.
            // ---------------------------------------------------------------

            std::cout
                << "Elite losses:\n";

            for (size_t i = 0;
                 i < population.size();
                 ++i)
            {
                std::cout
                    << "  "
                    << i + 1
                    << ": "
                    << population[i].loss
                    << '\n';
            }


            // ---------------------------------------------------------------
            // Exact global optimum for Hamming loss is 0.
            // ---------------------------------------------------------------

            if (best == 0.0)
            {
                std::cout
                    << "\n*** ZERO LOSS FOUND ***\n";

                break;
            }
        }


        // ====================================================================
        // Install best discovered model.
        // ====================================================================

        if (!population.empty())
        {
            copy_model_into_current(
                *population[0].model
            );

            std::cout
                << "\n========================================\n"
                << " Training finished\n"
                << " Best loss = "
                << population[0].loss
                << '\n'
                << "========================================\n";
        }
    }
};


// ============================================================================
// TRAINING DATA LOADER
// ============================================================================
//
// File format:
//
// [x0 x1 x2 x3 x4 x5 y0 y1]
// [x0 x1 x2 x3 x4 x5 y0 y1]
// ...
//
// Each value is int16_t.
//
// 6 values = 96-bit input
// 2 values = 32-bit target
//
// ============================================================================

inline bool load_training_data(
    const std::string& path,
    std::vector<TrainSample>& data)
{
    constexpr size_t INPUT_WORDS  = 6;
    constexpr size_t TARGET_WORDS = 2;
    constexpr size_t SAMPLE_WORDS = 8;

    std::ifstream file(
        path,
        std::ios::binary
    );

    if (!file)
    {
        std::cerr
            << "Failed to open training data: "
            << path
            << '\n';

        return false;
    }

    file.seekg(
        0,
        std::ios::end
    );

    const std::streamoff file_bytes =
        file.tellg();

    file.seekg(
        0,
        std::ios::beg
    );

    const size_t sample_bytes =
        SAMPLE_WORDS *
        sizeof(int16_t);

    if (file_bytes <= 0)
    {
        std::cerr
            << "Training data file is empty.\n";

        return false;
    }

    if (
        static_cast<size_t>(file_bytes)
        % sample_bytes != 0
    )
    {
        std::cerr
            << "Invalid training_data.bin size.\n"
            << "Each sample must contain exactly "
            << SAMPLE_WORDS
            << " int16 values.\n";

        return false;
    }

    const size_t sample_count =
        static_cast<size_t>(
            file_bytes /
            sample_bytes
        );

    data.clear();
    data.resize(sample_count);

    for (size_t i = 0;
         i < sample_count;
         ++i)
    {
        data[i].input.resize(
            INPUT_WORDS
        );

        data[i].target.resize(
            TARGET_WORDS
        );

        // ---------------------------------------------------------------
        // Input
        // ---------------------------------------------------------------

        for (size_t j = 0;
             j < INPUT_WORDS;
             ++j)
        {
            int16_t value = 0;

            file.read(
                reinterpret_cast<char*>(&value),
                sizeof(int16_t)
            );

            if (!file)
            {
                std::cerr
                    << "Error reading input at sample "
                    << i
                    << ".\n";

                return false;
            }

            data[i].input[j] =
                static_cast<uint16_t>(value);
        }

        // ---------------------------------------------------------------
        // Target
        // ---------------------------------------------------------------

        for (size_t j = 0;
             j < TARGET_WORDS;
             ++j)
        {
            int16_t value = 0;

            file.read(
                reinterpret_cast<char*>(&value),
                sizeof(int16_t)
            );

            if (!file)
            {
                std::cerr
                    << "Error reading target at sample "
                    << i
                    << ".\n";

                return false;
            }

            data[i].target[j] =
                static_cast<uint16_t>(value);
        }
    }

    std::cout
        << "Loaded "
        << sample_count
        << " training samples from "
        << path
        << '\n';

    return true;
}

} // namespace TNN