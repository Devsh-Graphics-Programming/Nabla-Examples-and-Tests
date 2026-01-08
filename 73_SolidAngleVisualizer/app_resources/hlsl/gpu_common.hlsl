#ifndef GPU_COMMON_HLSL
#define GPU_COMMON_HLSL

static const float32_t CIRCLE_RADIUS = 0.5f;

// --- Geometry Utils ---
struct ClippedSilhouette
{
    float32_t3 vertices[7]; // Max 7 vertices after clipping, unnormalized
    uint32_t count;
};

static const float32_t3 constCorners[8] = {
    float32_t3(-0.5f, -0.5f, -0.5f), float32_t3(0.5f, -0.5f, -0.5f), float32_t3(-0.5f, 0.5f, -0.5f), float32_t3(0.5f, 0.5f, -0.5f),
    float32_t3(-0.5f, -0.5f, 0.5f), float32_t3(0.5f, -0.5f, 0.5f), float32_t3(-0.5f, 0.5f, 0.5f), float32_t3(0.5f, 0.5f, 0.5f)};

static const uint32_t2 allEdges[12] = {
    {0, 1},
    {2, 3},
    {4, 5},
    {6, 7}, // X axis
    {0, 2},
    {1, 3},
    {4, 6},
    {5, 7}, // Y axis
    {0, 4},
    {1, 5},
    {2, 6},
    {3, 7}, // Z axis
};

// Maps face index (0-5) to its 4 corner indices in CCW order
static const uint32_t faceToCorners[6][4] = {
    {0, 2, 3, 1}, // Face 0: Z-
    {4, 5, 7, 6}, // Face 1: Z+
    {0, 4, 6, 2}, // Face 2: X-
    {1, 3, 7, 5}, // Face 3: X+
    {0, 1, 5, 4}, // Face 4: Y-
    {2, 6, 7, 3}  // Face 5: Y+
};

static float32_t3 corners[8];
static float32_t3 faceCenters[6] = {
    float32_t3(0, 0, 0), float32_t3(0, 0, 0), float32_t3(0, 0, 0),
    float32_t3(0, 0, 0), float32_t3(0, 0, 0), float32_t3(0, 0, 0)};

static const float32_t3 localNormals[6] = {
    float32_t3(0, 0, -1), // Face 0 (Z-)
    float32_t3(0, 0, 1),  // Face 1 (Z+)
    float32_t3(-1, 0, 0), // Face 2 (X-)
    float32_t3(1, 0, 0),  // Face 3 (X+)
    float32_t3(0, -1, 0), // Face 4 (Y-)
    float32_t3(0, 1, 0)   // Face 5 (Y+)
};

// TODO: unused, remove later
// Vertices are ordered CCW relative to the camera view.
static const uint32_t silhouettes[27][7] = {
    {6, 1, 3, 2, 6, 4, 5}, // 0: Black
    {6, 2, 6, 4, 5, 7, 3}, // 1: White
    {6, 0, 4, 5, 7, 3, 2}, // 2: Gray
    {6, 1, 3, 7, 6, 4, 5}, // 3: Red
    {4, 4, 5, 7, 6, 0, 0}, // 4: Green
    {6, 0, 4, 5, 7, 6, 2}, // 5: Blue
    {6, 0, 1, 3, 7, 6, 4}, // 6: Yellow
    {6, 0, 1, 5, 7, 6, 4}, // 7: Magenta
    {6, 0, 1, 5, 7, 6, 2}, // 8: Cyan
    {6, 1, 3, 2, 6, 7, 5}, // 9: Orange
    {4, 2, 6, 7, 3, 0, 0}, // 10: Light Orange
    {6, 0, 4, 6, 7, 3, 2}, // 11: Dark Orange
    {4, 1, 3, 7, 5, 0, 0}, // 12: Pink
    {6, 0, 4, 6, 7, 3, 2}, // 13: Light Pink
    {4, 0, 4, 6, 2, 0, 0}, // 14: Deep Rose
    {6, 0, 1, 3, 7, 5, 4}, // 15: Purple
    {4, 0, 1, 5, 4, 0, 0}, // 16: Light Purple
    {6, 0, 1, 5, 4, 6, 2}, // 17: Indigo
    {6, 0, 2, 6, 7, 5, 1}, // 18: Dark Green
    {6, 0, 2, 6, 7, 3, 1}, // 19: Lime
    {6, 0, 4, 6, 7, 3, 1}, // 20: Forest Green
    {6, 0, 2, 3, 7, 5, 1}, // 21: Navy
    {4, 0, 2, 3, 1, 0, 0}, // 22: Sky Blue
    {6, 0, 4, 6, 2, 3, 1}, // 23: Teal
    {6, 0, 2, 3, 7, 5, 4}, // 24: Brown
    {6, 0, 2, 3, 1, 5, 4}, // 25: Tan/Beige
    {6, 1, 5, 4, 6, 2, 3}  // 26: Dark Brown
};

// Binary packed silhouettes
static const uint32_t binSilhouettes[27] = {
    0b11000000000000101100110010011001,
    0b11000000000000011111101100110010,
    0b11000000000000010011111101100000,
    0b11000000000000101100110111011001,
    0b10000000000000000000110111101100,
    0b11000000000000010110111101100000,
    0b11000000000000100110111011001000,
    0b11000000000000100110111101001000,
    0b11000000000000010110111101001000,
    0b11000000000000101111110010011001,
    0b10000000000000000000011111110010,
    0b11000000000000010011111110100000,
    0b10000000000000000000101111011001,
    0b11000000000000010011111110100000,
    0b10000000000000000000010110100000,
    0b11000000000000100101111011001000,
    0b10000000000000000000100101001000,
    0b11000000000000010110100101001000,
    0b11000000000000001101111110010000,
    0b11000000000000001011111110010000,
    0b11000000000000001011111110100000,
    0b11000000000000001101111011010000,
    0b10000000000000000000001011010000,
    0b11000000000000001011010110100000,
    0b11000000000000100101111011010000,
    0b11000000000000100101001011010000,
    0b11000000000000011010110100101001,
};

uint32_t getSilhouetteVertex(uint32_t packedSil, uint32_t index)
{
    return (packedSil >> (3u * index)) & 0x7u;
}

// Get silhouette size
uint32_t getSilhouetteSize(uint32_t sil)
{
    return (sil >> 29u) & 0x7u;
}

// Check if vertex has negative z
bool getVertexZNeg(float32_t3x4 modelMatrix, uint32_t vertexIdx)
{
#if FAST
    float32_t3 localPos = float32_t3(
        (vertexIdx & 1) ? 0.5f : -0.5f,
        (vertexIdx & 2) ? 0.5f : -0.5f,
        (vertexIdx & 4) ? 0.5f : -0.5f);

    float32_t transformedZ = dot(modelMatrix[2].xyz, localPos) + modelMatrix[2].w;
    return transformedZ < 0.0f;
#else
    return corners[vertexIdx].z < 0.0f;
#endif
}

// Get world position of cube vertex
float32_t3 getVertex(float32_t3x4 modelMatrix, uint32_t vertexIdx)
{
#if FAST
    // Reconstruct local cube corner from index bits
    float32_t sx = (vertexIdx & 1) ? 0.5f : -0.5f;
    float32_t sy = (vertexIdx & 2) ? 0.5f : -0.5f;
    float32_t sz = (vertexIdx & 4) ? 0.5f : -0.5f;

    float32_t4x3 model = transpose(modelMatrix);

    // Transform to world
    // Full position, not just Z like getVertexZNeg
    return model[0].xyz * sx +
           model[1].xyz * sy +
           model[2].xyz * sz +
           model[3].xyz;
    // return mul(modelMatrix, float32_t4(sx, sy, sz, 1.0f));
#else
    return corners[vertexIdx];
#endif
}
#endif // GPU_COMMON_HLSL
