"""Quick test to compare GenericDecoder vs SteaneDecoder."""

import numpy as np
from src.qectostim.experiments.concatenated_css_v10 import GenericDecoder, create_concatenated_code
from src.qectostim.experiments.concatenated_css_v10_steane import create_steane_code, SteaneDecoder, create_concatenated_steane

# Create codes
steane_code = create_steane_code()
concat_generic = create_concatenated_code([steane_code, steane_code])
concat_steane = create_concatenated_steane(2)

# Create decoders
gen_dec = GenericDecoder(concat_generic)
steane_dec = SteaneDecoder(concat_steane)

print('Testing decode_measurement on various inputs:')
print()

# Test with various measurement patterns
test_patterns = [
    np.array([0,0,0,0,0,0,0]),  # No error
    np.array([1,0,0,0,0,0,0]),  # Error on qubit 0
    np.array([0,1,0,0,0,0,0]),  # Error on qubit 1
    np.array([0,0,1,0,0,0,0]),  # Error on qubit 2
    np.array([0,0,0,1,0,0,0]),  # Error on qubit 3
    np.array([0,0,0,0,1,0,0]),  # Error on qubit 4
    np.array([0,0,0,0,0,1,0]),  # Error on qubit 5
    np.array([0,0,0,0,0,0,1]),  # Error on qubit 6
    np.array([1,1,1,0,0,0,0]),  # Logical measurement = 1
    np.array([1,1,0,0,0,0,0]),  # Error on qubits 0,1 -> syndrome of qubit 2
]

all_match = True
for m in test_patterns:
    gen_x = gen_dec.decode_measurement(m, 'x')
    gen_z = gen_dec.decode_measurement(m, 'z')
    stean_x = steane_dec.decode_measurement(m, 'x')
    stean_z = steane_dec.decode_measurement(m, 'z')
    match = gen_x == stean_x and gen_z == stean_z
    status = "OK" if match else "MISMATCH"
    print(f'm={m.tolist()}: gen_x={gen_x}, steane_x={stean_x}, gen_z={gen_z}, steane_z={stean_z} [{status}]')
    if not match:
        all_match = False

print()
if all_match:
    print("All tests PASSED - decoders produce identical results")
else:
    print("Some tests FAILED - decoders differ")

# Also print syndrome table for debugging
print()
print("GenericDecoder syndrome table for Hz:")
print(gen_dec._syndrome_to_error_x)
