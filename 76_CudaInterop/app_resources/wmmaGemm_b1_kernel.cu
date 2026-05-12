#include <mma.h>
#include <cuda_runtime.h>

using namespace nvcuda;

// Define WMMA parameters
const int WMMA_M = 8;
const int WMMA_N = 8;
const int WMMA_K = 128;

extern "C" __global__ void b1_wmma_gemm_kernel(int* a, int* b, int* c, 
                                    int M, int N, int K) {
    // Leading dimensions
    int lda = K; 
    int ldb = K;
    int ldc = N;
    
    // Tile indices
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // Fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::experimental::precision::b1, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::experimental::precision::b1, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> acc_frag;
    
    // Initialize accumulator with zeros
    wmma::fill_fragment(acc_frag, 0);
    
    // Loop over the K-dimension
    for (int i = 0; i < K; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i / 32; // Indexing uint32_t
        
        int bRow = i / 32;
        int bCol = warpN * WMMA_N;
    
        // Load fragments
        // Note: load_matrix_sync handles the bit-packing layout internally
        wmma::load_matrix_sync(a_frag, a + (aRow * lda / 32 + aCol), lda);
        wmma::load_matrix_sync(b_frag, b + (bCol * ldb / 32 + bRow), ldb);
    
        // Perform XOR-Popcount MMA
        wmma::bmma_sync(acc_frag, a_frag, b_frag, acc_frag, wmma::experimental::bmmaBitOpAND);
    }
    
    // Store the result
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    int* outputLoc = c + (cRow * ldc + cCol);
    wmma::store_matrix_sync(outputLoc, acc_frag, ldc, wmma::mem_row_major);

}
