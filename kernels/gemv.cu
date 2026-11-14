// CUDA source reference for gemv_q8_0 and gemv_fused_ffn_q8_0
// Compiled to PTX with warp shuffle reduction and zero shared-memory overhead.

struct BlockQ8_0 {
    unsigned short d; // half precision scale
    signed char qs[32];
};

extern "C" __global__ void gemv_q8_0(
    const BlockQ8_0 *weights,
    const float *input,
    float *output,
    unsigned int cols,
    unsigned int rows
) {
    unsigned int lane = threadIdx.x;
    unsigned int warp_id = threadIdx.y;
    unsigned int row = blockIdx.x * blockDim.y + warp_id;
    if (row >= rows) return;

    unsigned int num_blocks = cols / 32;
    const BlockQ8_0 *row_weights = weights + (size_t)row * num_blocks;

    float sum = 0.0f;
    for (unsigned int b = 0; b < num_blocks; ++b) {
        float scale = __half2float(*(const __half *)&row_weights[b].d);
        float q = (float)row_weights[b].qs[lane];
        float act = input[b * 32 + lane];
        sum += q * scale * act;
    }

    // Warp shuffle reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        output[row] = sum;
    }
}
