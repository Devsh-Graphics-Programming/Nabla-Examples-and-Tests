#ifndef _SAMPLING_HLSL_
#define _SAMPLING_HLSL_

// Include the spherical triangle utilities
#include <gpu_common.hlsl>
#include <nbl/builtin/hlsl/shapes/spherical_triangle.hlsl>
#include <nbl/builtin/hlsl/sampling/spherical_triangle.hlsl>
#include <nbl/builtin/hlsl/sampling/projected_spherical_triangle.hlsl>
#include "nbl/builtin/hlsl/random/pcg.hlsl"
#include "nbl/builtin/hlsl/random/xoroshiro.hlsl"

using namespace nbl::hlsl;
// Sampling mode enum
#define SAMPLING_MODE_SOLID_ANGLE 0
#define SAMPLING_MODE_PROJECTED_SOLID_ANGLE 1

// Maximum number of triangles we can have after clipping
// Without clipping, max 3 faces can be visible at once so 3 faces * 2 triangles = 6 edges, forming max 4 triangles
// With clipping, one more edge. 7 - 2 = 5 max triangles because fanning from one vertex
#define MAX_TRIANGLES 5

// Minimal cached sampling data - only what's needed for selection
struct SamplingData
{
    uint32_t count;                               // Number of valid triangles
    uint32_t samplingMode;                        // Mode used during build
    float32_t totalWeight;                        // Sum of all triangle weights
    float32_t3 faceNormal;                        // Face normal (only used for projected mode)
    float32_t triangleSolidAngles[MAX_TRIANGLES]; // Weight per triangle (for selection)
    uint32_t triangleIndices[MAX_TRIANGLES];      // Vertex index i (forms triangle with v0, vi, vi+1)
};

float32_t2 nextRandomUnorm2(inout nbl::hlsl::Xoroshiro64StarStar rnd)
{
    return float32_t2(
        float32_t(rnd()) * 2.3283064365386963e-10,
        float32_t(rnd()) * 2.3283064365386963e-10);
}

float32_t computeProjectedSolidAngleFallback(float32_t3 v0, float32_t3 v1, float32_t3 v2, float32_t3 N)
{
    // 1. Get edge normals (unit vectors)
    // We use the cross product of the vertices (unit vectors on sphere)
    float32_t3 n0 = cross(v0, v1);
    float32_t3 n1 = cross(v1, v2);
    float32_t3 n2 = cross(v2, v0);

    // 2. Normalize edge normals (magnitude is sin of the arc length)
    float32_t l0 = length(n0);
    float32_t l1 = length(n1);
    float32_t l2 = length(n2);

    // Guard against degenerate triangles
    if (l0 < 1e-7 || l1 < 1e-7 || l2 < 1e-7)
        return 0.0f;

    n0 /= l0;
    n1 /= l1;
    n2 /= l2;

    // 3. Get arc lengths (angles in radians)
    float32_t a = asin(clamp(l0, -1.0, 1.0)); // side v0-v1
    float32_t b = asin(clamp(l1, -1.0, 1.0)); // side v1-v2
    float32_t c = asin(clamp(l2, -1.0, 1.0)); // side v2-v0

    // Handle acos/asin quadrant if dot product is negative
    if (dot(v0, v1) < 0)
        a = 3.14159265 - a;
    if (dot(v1, v2) < 0)
        b = 3.14159265 - b;
    if (dot(v2, v0) < 0)
        c = 3.14159265 - c;

    // 4. Compute projected solid angle
    float32_t Gamma = 0.5f * (a * dot(n0, N) + b * dot(n1, N) + c * dot(n2, N));

    // Return the absolute value of the total
    return abs(Gamma);
}

// Build sampling data once - cache only weights for triangle selection
SamplingData buildSamplingDataFromSilhouette(ClippedSilhouette silhouette, uint32_t samplingMode)
{
    SamplingData data;
    data.count = 0;
    data.totalWeight = 0.0f;
    data.samplingMode = samplingMode;
    data.faceNormal = float32_t3(0, 0, 0);

    if (silhouette.count < 3)
        return data;

    const float32_t3 v0 = silhouette.vertices[0];
    const float32_t3 origin = float32_t3(0, 0, 0);

    // Compute face normal ONCE before the loop - silhouette is planar!
    if (samplingMode == SAMPLING_MODE_PROJECTED_SOLID_ANGLE)
    {
        float32_t3 v1 = silhouette.vertices[1];
        float32_t3 v2 = silhouette.vertices[2];
        data.faceNormal = normalize(cross(v1 - v0, v2 - v0));
    }

    // Build fan triangulation from v0
    NBL_UNROLL
    for (uint32_t i = 1; i < silhouette.count - 1; i++)
    {
        float32_t3 v1 = silhouette.vertices[i];
        float32_t3 v2 = silhouette.vertices[i + 1];

        shapes::SphericalTriangle<float32_t> shapeTri = shapes::SphericalTriangle<float32_t>::create(v0, v1, v2, origin);

        // Skip degenerate triangles
        if (shapeTri.pyramidAngles())
            continue;

        // Calculate triangle solid angle
        float32_t solidAngle;
        if (samplingMode == SAMPLING_MODE_PROJECTED_SOLID_ANGLE)
        {
            // scalar_type projectedSolidAngleOfTriangle(const vector3_type receiverNormal, NBL_REF_ARG(vector3_type) cos_sides, NBL_REF_ARG(vector3_type) csc_sides, NBL_REF_ARG(vector3_type) cos_vertices)
            float32_t3 cos_vertices = clamp(
                (shapeTri.cos_sides - shapeTri.cos_sides.yzx * shapeTri.cos_sides.zxy) *
                    shapeTri.csc_sides.yzx * shapeTri.csc_sides.zxy,
                float32_t3(-1.0f, -1.0f, -1.0f),
                float32_t3(1.0f, 1.0f, 1.0f));
            solidAngle = shapeTri.projectedSolidAngleOfTriangle(data.faceNormal, shapeTri.cos_sides, shapeTri.csc_sides, cos_vertices);
        }
        else
        {
            solidAngle = shapeTri.solidAngleOfTriangle();
        }

        if (solidAngle <= 0.0f)
            continue;

        // Store only what's needed for weighted selection
        data.triangleSolidAngles[data.count] = solidAngle;
        data.triangleIndices[data.count] = i;
        data.totalWeight += solidAngle;
        data.count++;
    }

#ifdef DEBUG_DATA
    // Validate no antipodal edges exist (would create spherical lune)
    for (uint32_t i = 0; i < silhouette.count; i++)
    {
        uint32_t j = (i + 1) % silhouette.count;
        float32_t3 n1 = normalize(silhouette.vertices[i]);
        float32_t3 n2 = normalize(silhouette.vertices[j]);

        if (dot(n1, n2) < -0.99f)
        {
            DebugDataBuffer[0].sphericalLuneDetected = 1;
            assert(false && "Spherical lune detected: antipodal silhouette edge");
        }
    }
    DebugDataBuffer[0].maxTrianglesExceeded = (data.count > MAX_TRIANGLES);

    DebugDataBuffer[0].clippedSilhouetteVertexCount = silhouette.count;
    for (uint32_t v = 0; v < silhouette.count; v++)
    {
        DebugDataBuffer[0].clippedSilhouetteVertices[v] = silhouette.vertices[v];
    }

    DebugDataBuffer[0].triangleCount = data.count;
    DebugDataBuffer[0].totalSolidAngles = data.totalWeight;
    for (uint32_t tri = 0; tri < data.count; tri++)
    {
        DebugDataBuffer[0].solidAngles[tri] = data.triangleSolidAngles[tri];
    }
#endif

    return data;
}

// Sample using cached selection weights, but recompute geometry on-demand
float32_t3 sampleFromData(SamplingData data, ClippedSilhouette silhouette, float32_t2 xi, out float32_t pdf, out uint32_t selectedIdx)
{
    selectedIdx = 0;

    // Handle empty or invalid data
    if (data.count == 0 || data.totalWeight <= 0.0f)
    {
        pdf = 0.0f;
        return float32_t3(0, 0, 1);
    }

    // Select triangle using cached weighted random selection
    float32_t targetWeight = xi.x * data.totalWeight;
    float32_t cumulativeWeight = 0.0f;
    float32_t prevCumulativeWeight = 0.0f;

    NBL_UNROLL
    for (uint32_t i = 0; i < data.count; i++)
    {
        prevCumulativeWeight = cumulativeWeight;
        cumulativeWeight += data.triangleSolidAngles[i];

        if (targetWeight <= cumulativeWeight)
        {
            selectedIdx = i;
            break;
        }
    }

    // Remap xi.x to [0,1] within selected triangle's solidAngle interval
    float32_t triSolidAngle = data.triangleSolidAngles[selectedIdx];
    float32_t u = (targetWeight - prevCumulativeWeight) / max(triSolidAngle, 1e-7f);

    // Reconstruct the selected triangle geometry
    uint32_t vertexIdx = data.triangleIndices[selectedIdx];
    float32_t3 v0 = silhouette.vertices[0];
    float32_t3 v1 = silhouette.vertices[vertexIdx];
    float32_t3 v2 = silhouette.vertices[vertexIdx + 1];

	float32_t3 faceNormal = normalize(cross(v1 - v0, v2 - v0));

    float32_t3 origin = float32_t3(0, 0, 0);

    shapes::SphericalTriangle<float32_t> shapeTri = shapes::SphericalTriangle<float32_t>::create(v0, v1, v2, origin);

    // Compute vertex angles once
    float32_t3 cos_vertices = clamp(
        (shapeTri.cos_sides - shapeTri.cos_sides.yzx * shapeTri.cos_sides.zxy) *
            shapeTri.csc_sides.yzx * shapeTri.csc_sides.zxy,
        float32_t3(-1.0f, -1.0f, -1.0f),
        float32_t3(1.0f, 1.0f, 1.0f));
    float32_t3 sin_vertices = sqrt(float32_t3(1.0f, 1.0f, 1.0f) - cos_vertices * cos_vertices);

    // Sample based on mode
    float32_t3 direction;
    float32_t rcpPdf;

    if (data.samplingMode == SAMPLING_MODE_PROJECTED_SOLID_ANGLE)
    {
        sampling::ProjectedSphericalTriangle<float32_t> samplingTri =
            sampling::ProjectedSphericalTriangle<float32_t>::create(shapeTri);

        direction = samplingTri.generate(
            rcpPdf,
            triSolidAngle,
            cos_vertices,
            sin_vertices,
            shapeTri.cos_sides[0],
            shapeTri.cos_sides[2],
            shapeTri.csc_sides[1],
            shapeTri.csc_sides[2],
            faceNormal,
            false,
            float32_t2(u, xi.y));
        triSolidAngle = rcpPdf; // projected solid angle returned as rcpPdf
    }
    else
    {
        sampling::SphericalTriangle<float32_t> samplingTri =
            sampling::SphericalTriangle<float32_t>::create(shapeTri);

        direction = samplingTri.generate(
            triSolidAngle,
            cos_vertices,
            sin_vertices,
            shapeTri.cos_sides[0],
            shapeTri.cos_sides[2],
            shapeTri.csc_sides[1],
            shapeTri.csc_sides[2],
            float32_t2(u, xi.y));
    }

    // Calculate PDF
    float32_t trianglePdf = 1.0f / triSolidAngle;
    float32_t selectionProb = triSolidAngle / data.totalWeight;
    pdf = trianglePdf * selectionProb;

    return normalize(direction);
}

#if VISUALIZE_SAMPLES

float32_t4 visualizeSamples(float32_t2 screenUV, float32_t3 spherePos, ClippedSilhouette silhouette,
                            uint32_t samplingMode, uint32_t frameIndex, SamplingData samplingData, uint32_t numSamples, inout RWStructuredBuffer<ResultData> DebugDataBuffer)
{
    float32_t4 accumColor = 0;

    if (silhouette.count == 0)
        return 0;

    float32_t2 pssSize = float32_t2(0.3, 0.3);  // 30% of screen
    float32_t2 pssPos = float32_t2(0.01, 0.01); // Offset from corner
    bool isInsidePSS = all(and(screenUV >= pssPos, screenUV <= (pssPos + pssSize)));

    DebugDataBuffer[0].sampleCount = numSamples;
    for (uint32_t i = 0; i < numSamples; i++)
    {
        nbl::hlsl::random::PCG32 seedGen = nbl::hlsl::random::PCG32::construct(frameIndex * 65536u + i);
        const uint32_t seed1 = seedGen();
        const uint32_t seed2 = seedGen();
        nbl::hlsl::Xoroshiro64StarStar rnd = nbl::hlsl::Xoroshiro64StarStar::construct(uint32_t2(seed1, seed2));
        float32_t2 xi = nextRandomUnorm2(rnd);

        float32_t pdf;
        uint32_t triIdx;
        float32_t3 sampleDir = sampleFromData(samplingData, silhouette, xi, pdf, triIdx);

        DebugDataBuffer[0].rayData[i] = float32_t4(sampleDir, pdf);

        float32_t dist3D = distance(sampleDir, normalize(spherePos));
        float32_t alpha3D = 1.0f - smoothstep(0.0f, 0.02f, dist3D);

        if (alpha3D > 0.0f && !isInsidePSS)
        {
            float32_t3 sampleColor = colorLUT[triIdx].rgb;
            accumColor += float32_t4(sampleColor * alpha3D, alpha3D);
        }

        if (isInsidePSS)
        {
            // Map the raw xi to the PSS square dimensions
            float32_t2 xiPixelPos = pssPos + xi * pssSize;
            float32_t dist2D = distance(screenUV, xiPixelPos);

            float32_t alpha2D = drawCross2D(screenUV, xiPixelPos, 0.005f, 0.001f);
            if (alpha2D > 0.0f)
            {
                float32_t3 sampleColor = colorLUT[triIdx].rgb;
                accumColor += float32_t4(sampleColor * alpha2D, alpha2D);
            }
        }
    }

    // just the outline of the PSS
    if (isInsidePSS && accumColor.a < 0.1)
        accumColor = float32_t4(0.1, 0.1, 0.1, 1.0);

    return accumColor;
}
#endif
#endif
