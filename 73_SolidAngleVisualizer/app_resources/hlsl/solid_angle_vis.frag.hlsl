//// Copyright (C) 2026-2026 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#pragma wave shader_stage(fragment)

#include "common.hlsl"
#include <nbl/builtin/hlsl/ext/FullScreenTriangle/SVertexAttributes.hlsl>

using namespace nbl::hlsl;
using namespace ext::FullScreenTriangle;

#include "drawing.hlsl"
#include "utils.hlsl"
#include "silhouette.hlsl"
#include "triangle_sampling.hlsl"
#include "pyramid_sampling.hlsl"
#include "parallelogram_sampling.hlsl"

[[vk::push_constant]] struct PushConstants pc;

static const SAMPLING_MODE samplingMode = (SAMPLING_MODE)SAMPLING_MODE_CONST;

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

#if VISUALIZE_SAMPLES
float32_t4 visualizeSample(float32_t3 sampleDir, float32_t2 xi, uint32_t index, float32_t2 screenUV, float32_t3 spherePos, float32_t2 ndc, float32_t aaWidth
#if DEBUG_DATA
                           ,
                           inout RWStructuredBuffer<ResultData> DebugDataBuffer
#endif
)
{
    float32_t4 accumColor = 0;

    float32_t2 pssSize = float32_t2(0.3, 0.3);  // 30% of screen
    float32_t2 pssPos = float32_t2(0.01, 0.01); // Offset from corner
    bool isInsidePSS = all(and(screenUV >= pssPos, screenUV <= (pssPos + pssSize)));

    float32_t dist3D = distance(sampleDir, normalize(spherePos));
    float32_t alpha3D = 1.0f - smoothstep(0.0f, 0.02f, dist3D);

    if (alpha3D > 0.0f /* && !isInsidePSS*/)
    {
        float32_t3 sampleColor = colorLUT[index].rgb;
        accumColor += float32_t4(sampleColor * alpha3D, alpha3D);
    }

    // if (isInsidePSS)
    // {
    // 	// Map the raw xi to the PSS square dimensions
    // 	float32_t2 xiPixelPos = pssPos + xi * pssSize;
    // 	float32_t dist2D = distance(screenUV, xiPixelPos);

    // 	float32_t alpha2D = drawCross2D(screenUV, xiPixelPos, 0.005f, 0.001f);
    // 	if (alpha2D > 0.0f)
    // 	{
    // 		float32_t3 sampleColor = colorLUT[index].rgb;
    // 		accumColor += float32_t4(sampleColor * alpha2D, alpha2D);
    // 	}
    // }

    // // just the outline of the PSS
    // if (isInsidePSS && accumColor.a < 0.1)
    // 	accumColor = float32_t4(0.1, 0.1, 0.1, 1.0);

    return accumColor;
}
#endif // VISUALIZE_SAMPLES

// [shader("pixel")]
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
        uint32_t sil = ClippedSilhouette::computeRegionAndConfig(pc.modelMatrix, region, configIndex, vertexCount);

        uint32_t silEdgeMask = 0; // TODO: take from 'fast' compute()
#if DEBUG_DATA
        validateSilhouetteEdges(sil, vertexCount, silEdgeMask);
#endif
        ClippedSilhouette silhouette;
        silhouette.compute(pc.modelMatrix, vertexCount, sil);

#if VISUALIZE_SAMPLES
        // Draw silhouette edges on the sphere
        for (uint32_t ei = 0; ei < silhouette.count; ei++)
        {
            float32_t3 v0 = normalize(silhouette.vertices[ei]);
            float32_t3 v1 = normalize(silhouette.vertices[(ei + 1) % silhouette.count]);
            float32_t3 pts[2] = {v0, v1};
            color += drawEdge(0, pts, spherePos, aaWidth);
        }
#endif

        TriangleFanSampler samplingData;
        Parallelogram parallelogram;
        SphericalPyramid pyramid;
        UrenaSampler urena;
        BiquadraticSampler biquad;
        BilinearSampler bilin;

        SilEdgeNormals silEdgeNormals;
        //=====================================================================
        // Building
        //=====================================================================
        if (samplingMode == SAMPLING_MODE::TRIANGLE_SOLID_ANGLE ||
            samplingMode == SAMPLING_MODE::TRIANGLE_PROJECTED_SOLID_ANGLE)
        {
            samplingData = TriangleFanSampler::create(silhouette, samplingMode);
        }
        else if (samplingMode == SAMPLING_MODE::PROJECTED_PARALLELOGRAM_SOLID_ANGLE)
        {
            silhouette.normalize();
            parallelogram = Parallelogram::create(silhouette, silEdgeNormals
#if VISUALIZE_SAMPLES
                                                  ,
                                                  ndc, spherePos, aaWidth, color
#endif
            );
        }
        else if (samplingMode == SAMPLING_MODE::SYMMETRIC_PYRAMID_SOLID_ANGLE_RECTANGLE ||
                 samplingMode == SAMPLING_MODE::SYMMETRIC_PYRAMID_SOLID_ANGLE_BIQUADRATIC ||
                 samplingMode == SAMPLING_MODE::SYMMETRIC_PYRAMID_SOLID_ANGLE_BILINEAR)
        {
            pyramid = SphericalPyramid::create(silhouette, silEdgeNormals
#if VISUALIZE_SAMPLES
                                               ,
                                               ndc, spherePos, aaWidth, color
#endif
            );

            if (samplingMode == SAMPLING_MODE::SYMMETRIC_PYRAMID_SOLID_ANGLE_RECTANGLE)
                urena = UrenaSampler::create(pyramid);
            else if (samplingMode == SAMPLING_MODE::SYMMETRIC_PYRAMID_SOLID_ANGLE_BIQUADRATIC)
                biquad = BiquadraticSampler::create(pyramid);
            else if (samplingMode == SAMPLING_MODE::SYMMETRIC_PYRAMID_SOLID_ANGLE_BILINEAR)
                bilin = BilinearSampler::create(pyramid);
        }

#if DEBUG_DATA
        uint32_t validSampleCount = 0u;
        DebugDataBuffer[0].sampleCount = pc.sampleCount;
#endif
        //=====================================================================
        // Sampling
        //=====================================================================
        for (uint32_t i = 0; i < pc.sampleCount; i++)
        {
            // Hash the invocation to offset the grid
            float32_t2 xi = float32_t2(
                (float32_t(i & 7u) + 0.5) / 8.0f,
                (float32_t(i >> 3u) + 0.5) / 8.0f);

            float32_t pdf;
            uint32_t index = 0;
            float32_t3 sampleDir;
            bool valid;

            if (samplingMode == SAMPLING_MODE::TRIANGLE_SOLID_ANGLE || samplingMode == SAMPLING_MODE::TRIANGLE_PROJECTED_SOLID_ANGLE)
                sampleDir = samplingData.sample(silhouette, xi, pdf, index);
            else if (samplingMode == SAMPLING_MODE::PROJECTED_PARALLELOGRAM_SOLID_ANGLE)
                sampleDir = parallelogram.sample(silEdgeNormals, xi, pdf, valid);
            else if (samplingMode == SAMPLING_MODE::SYMMETRIC_PYRAMID_SOLID_ANGLE_RECTANGLE)
                sampleDir = urena.sample(pyramid, silEdgeNormals, xi, pdf, valid);
            else if (samplingMode == SAMPLING_MODE::SYMMETRIC_PYRAMID_SOLID_ANGLE_BIQUADRATIC)
                sampleDir = biquad.sample(pyramid, silEdgeNormals, xi, pdf, valid);
            else if (samplingMode == SAMPLING_MODE::SYMMETRIC_PYRAMID_SOLID_ANGLE_BILINEAR)
                sampleDir = bilin.sample(pyramid, silEdgeNormals, xi, pdf, valid);

            if (!valid)
            {
                pdf = 0.0f;
                // sampleDir = float32_t3(0, 0, 1);
            }
#if DEBUG_DATA
            else
            {
                validSampleCount++;
            }

            DebugDataBuffer[0].rayData[i] = float32_t4(sampleDir, pdf);
#endif

#if VISUALIZE_SAMPLES
            // Draw samples on sphere
            color += visualizeSample(sampleDir, xi, index, vx.uv, spherePos, ndc, aaWidth
#if DEBUG_DATA
                                     ,
                                     DebugDataBuffer
#endif
            );
#else
            if (pdf > 0.0f)
                color += float4(sampleDir * 0.02f / pdf, 1.0f);
#endif // VISUALIZE_SAMPLES
        }

#if VISUALIZE_SAMPLES

        // For debugging: Draw a small indicator of which faces are found
        // color += drawVisibleFaceOverlay(pc.modelMatrix, spherePos, region, aaWidth);

        // color += drawFaces(pc.modelMatrix, spherePos, aaWidth);

        // Draw clipped silhouette vertices
        // color += drawClippedSilhouetteVertices(ndc, silhouette, aaWidth);
        // color += drawHiddenEdges(pc.modelMatrix, spherePos, silEdgeMask, aaWidth);
        // color += drawCorners(pc.modelMatrix, ndc, aaWidth, 0.05f);
        color += drawRing(ndc, aaWidth);

        if (all(vx.uv >= float32_t2(0.f, 0.97f)) && all(vx.uv <= float32_t2(0.03f, 1.0f)))
        {
            return float32_t4(colorLUT[configIndex], 1.0f);
        }
#else
#endif // VISUALIZE_SAMPLES

#if DEBUG_DATA
        InterlockedAdd(DebugDataBuffer[0].validSampleCount, validSampleCount);
        InterlockedAdd(DebugDataBuffer[0].threadCount, 1u);
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

    return color;
}
