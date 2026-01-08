#pragma wave shader_stage(fragment)

#include "common.hlsl"
#include <nbl/builtin/hlsl/ext/FullScreenTriangle/SVertexAttributes.hlsl>

using namespace nbl::hlsl;
using namespace ext::FullScreenTriangle;

[[vk::binding(0, 0)]] RWStructuredBuffer<ResultData> DebugDataBuffer; // TODO: move below other includes

#define VISUALIZE_SAMPLES 1

#include "utils.hlsl"
#include "Drawing.hlsl"
#include "Sampling.hlsl"
#include "silhouette.hlsl"
[[vk::push_constant]] struct PushConstants pc;

void setDebugData(uint32_t sil, uint32_t3 region, uint32_t configIndex)
{
#if DEBUG_DATA
    DebugDataBuffer[0].region = uint32_t3(region);
    DebugDataBuffer[0].silhouetteIndex = uint32_t(configIndex);
    DebugDataBuffer[0].silhouetteVertexCount = uint32_t(getSilhouetteSize(sil));
    for (uint32_t i = 0; i < 6; i++)
    {
        DebugDataBuffer[0].vertices[i] = uint32_t(getSilhouetteVertex(sil, i));
    }
    DebugDataBuffer[0].silhouette = sil;
#endif
}

void computeCubeGeo()
{
    for (uint32_t i = 0; i < 8; i++)
        corners[i] = mul(pc.modelMatrix, float32_t4(constCorners[i], 1.0f)).xyz;

    for (uint32_t f = 0; f < 6; f++)
    {
        faceCenters[f] = float32_t3(0, 0, 0);
        for (uint32_t v = 0; v < 4; v++)
            faceCenters[f] += corners[faceToCorners[f][v]];
        faceCenters[f] /= 4.0f;
    }
}

void validateSilhouetteEdges(uint32_t sil, uint32_t vertexCount, inout uint32_t silEdgeMask)
{
#if DEBUG_DATA
    {
        for (uint32_t i = 0; i < vertexCount; i++)
        {
            uint32_t vIdx = i % vertexCount;
            uint32_t v1Idx = (i + 1) % vertexCount;

            uint32_t v0Corner = getSilhouetteVertex(sil, vIdx);
            uint32_t v1Corner = getSilhouetteVertex(sil, v1Idx);
            // Mark edge as part of silhouette
            for (uint32_t e = 0; e < 12; e++)
            {
                uint32_t2 edge = allEdges[e];
                if ((edge.x == v0Corner && edge.y == v1Corner) ||
                    (edge.x == v1Corner && edge.y == v0Corner))
                {
                    silEdgeMask |= (1u << e);
                }
            }
        }
        validateEdgeVisibility(pc.modelMatrix, sil, vertexCount, silEdgeMask);
    }
#endif
}

void computeSpherePos(SVertexAttributes vx, out float32_t2 ndc, out float32_t3 spherePos)
{
    ndc = vx.uv * 2.0f - 1.0f;
    float32_t aspect = pc.viewport.z / pc.viewport.w;
    ndc.x *= aspect;

    float32_t2 normalized = ndc / CIRCLE_RADIUS;
    float32_t r2 = dot(normalized, normalized);

    if (r2 <= 1.0f)
    {
        spherePos = float32_t3(normalized.x, normalized.y, sqrt(1.0f - r2));
    }
    else
    {
        float32_t uv2Plus1 = r2 + 1.0f;
        spherePos = float32_t3(normalized.x * 2.0f, normalized.y * 2.0f, 1.0f - r2) / uv2Plus1;
    }
    spherePos = normalize(spherePos);
}

[[vk::location(0)]] float32_t4 main(SVertexAttributes vx) : SV_Target0
{
    float32_t4 color = float32_t4(0, 0, 0, 0);
    for (uint32_t i = 0; i < 1; i++)
    {
        float32_t aaWidth = length(float32_t2(ddx(vx.uv.x), ddy(vx.uv.y)));
        float32_t3 spherePos;
        float32_t2 ndc;
        computeSpherePos(vx, ndc, spherePos);
#if !FAST || DEBUG_DATA
        computeCubeGeo();
#endif
        uint32_t3 region;
        uint32_t configIndex;
        uint32_t vertexCount;
        uint32_t sil = computeRegionAndConfig(pc.modelMatrix, region, configIndex, vertexCount);

        uint32_t silEdgeMask = 0; // TODO: take from 'fast' computeSilhouette()
#if DEBUG_DATA
        validateSilhouetteEdges(sil, vertexCount, silEdgeMask);
#endif
        ClippedSilhouette silhouette;

#if VISUALIZE_SAMPLES
        color += computeSilhouette(pc.modelMatrix, vertexCount, sil, spherePos, aaWidth, silhouette);
#else
        computeSilhouette(pc.modelMatrix, vertexCount, sil, silhouette);
#endif
        // Draw clipped silhouette vertices
        // color += drawClippedSilhouetteVertices(ndc, silhouette, aaWidth);

        SamplingData samplingData = buildSamplingDataFromSilhouette(silhouette, pc.samplingMode);
#if VISUALIZE_SAMPLES

        // For debugging: Draw a small indicator of which faces are found
        // color += drawVisibleFaceOverlay(pc.modelMatrix, spherePos, region, aaWidth);

        // color += drawFaces(pc.modelMatrix, spherePos, aaWidth);

        // Draw samples on sphere
        color += visualizeSamples(vx.uv, spherePos, silhouette, pc.samplingMode, pc.frameIndex, samplingData, 64, DebugDataBuffer);

        color += drawHiddenEdges(pc.modelMatrix, spherePos, silEdgeMask, aaWidth);
        color += drawCorners(pc.modelMatrix, ndc, aaWidth);
        color += drawRing(ndc, aaWidth);

        if (all(vx.uv >= float32_t2(0.49f, 0.49f)) && all(vx.uv <= float32_t2(0.51f, 0.51f)))
        {
            return float32_t4(colorLUT[configIndex], 1.0f);
        }
#else
        nbl::hlsl::random::PCG32 seedGen = nbl::hlsl::random::PCG32::construct(65536u + i);
        const uint32_t2 seeds = uint32_t2(seedGen(), seedGen());
        nbl::hlsl::Xoroshiro64StarStar rnd = nbl::hlsl::Xoroshiro64StarStar::construct(seeds);
        float32_t2 xi = nextRandomUnorm2(rnd);

        float32_t pdf;
        uint32_t triIdx;
        float32_t3 sampleDir = sampleFromData(samplingData, silhouette, xi, pdf, triIdx);

        color += float4(sampleDir * 0.02f / pdf, 1.0f);
#endif // VISUALIZE_SAMPLES
        setDebugData(sil, region, configIndex);
    }

    return color;
}