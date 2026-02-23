// Common definitions for Metal shaders
#ifndef COMMON_H
#define COMMON_H

// Thread group size for tiling operations
#define TILE_SIZE 16

// Maximum vector size for vectorized operations
#define VECTOR_SIZE 4

// Numerical stability constants
#define EPSILON 1e-7f
#define LOG_EPSILON log(EPSILON)

// Activation function coefficients
#define SIGMOID_COEFF 0.197f

// Memory alignment for optimal performance
#define ALIGNMENT 16

#endif // COMMON_H
