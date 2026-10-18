import numpy as np
import os

def generate_binary_files():
    # 1. pos_weights.bin: 6144 random int16
    pos_weights = np.random.randint(-32768, 32767, size=6144, dtype=np.int16)
    pos_weights.tofile('Omegus_mc/pos_weights.bin')
    print(f"Generated pos_weights.bin: {os.path.getsize('Omegus_mc/pos_weights.bin')} bytes")

    # 2. neg_weights.bin: 6144 random int16
    neg_weights = np.random.randint(-32768, 32767, size=6144, dtype=np.int16)
    neg_weights.tofile('Omegus_mc/neg_weights.bin')
    print(f"Generated neg_weights.bin: {os.path.getsize('Omegus_mc/neg_weights.bin')} bytes")

    # 3. biases.bin: 544 random float32
    # Using uniform distribution between -1.0 and 1.0
    biases = np.random.uniform(-1.0, 1.0, size=544).astype(np.float32)
    biases.tofile('Omegus_mc/biases.bin')
    print(f"Generated biases.bin: {os.path.getsize('Omegus_mc/biases.bin')} bytes")

    # 4. inputs.bin: 6 random int16
    inputs = np.random.randint(-32768, 32767, size=6000, dtype=np.int16)
    inputs.tofile('Omegus_mc/inputs.bin')
    print(f"Generated inputs.bin: {os.path.getsize('Omegus_mc/inputs.bin')} bytes")

    # 5. training_data.bin: 800 random int16
    training_data = np.random.randint(-32768, 32767, size=16000, dtype=np.int16)
    training_data.tofile('Omegus_mc/training_data.bin')
    print(f"Generated training_data.bin: {os.path.getsize('Omegus_mc/training_data.bin')} bytes")

    # 6. latent_weight.bin: 98304 float32 values, all 0
    latent_weight = np.zeros(98304, dtype=np.float32)
    latent_weight.tofile('Omegus_mc/latent_weight.bin')
    print(f"Generated latent_weight.bin: {os.path.getsize('Omegus_mc/latent_weight.bin')} bytes")

if __name__ == "__main__":
    generate_binary_files()
    print("\nAll files generated successfully.")