//// Copyright (C) 2026-2026 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _SOLID_ANGLE_VIS_EXAMPLE_COMMON_HLSL_INCLUDED_
#define _SOLID_ANGLE_VIS_EXAMPLE_COMMON_HLSL_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

#define FAST 1

namespace nbl
{
    namespace hlsl
    {
        // Sampling mode enum
        enum SAMPLING_MODE : uint32_t
        {
            TRIANGLE_SOLID_ANGLE,
            TRIANGLE_PROJECTED_SOLID_ANGLE,
            PROJECTED_PARALLELOGRAM_SOLID_ANGLE,
            SYMMETRIC_PYRAMID_SOLID_ANGLE_RECTANGLE,
            SYMMETRIC_PYRAMID_SOLID_ANGLE_BIQUADRATIC,
            SYMMETRIC_PYRAMID_SOLID_ANGLE_BILINEAR,
            Count
        };

        struct ResultData
        {
            // Silhouette
            uint32_t3 region;
            uint32_t silhouetteIndex;
            uint32_t silhouetteVertexCount;
            uint32_t silhouette;
            uint32_t positiveVertCount;
            uint32_t edgeVisibilityMismatch;
            uint32_t clipMask;
            uint32_t clipCount;
            uint32_t rotatedSil;
            uint32_t wrapAround;
            uint32_t rotatedClipMask;
            uint32_t rotateAmount;
            uint32_t vertices[6];
            uint32_t clippedSilhouetteVertexCount;
            float32_t3 clippedSilhouetteVertices[7];
            uint32_t clippedSilhouetteVerticesIndices[7];

            // Parallelogram
            uint32_t parallelogramDoesNotBound;
            float32_t parallelogramArea;
            uint32_t failedVertexIndex;
            uint32_t edgeIsConvex[4];
            uint32_t parallelogramVerticesInside;
            uint32_t parallelogramEdgesInside;
            float32_t2 parallelogramCorners[4];

            // spherical triangle
            uint32_t maxTrianglesExceeded;
            uint32_t sphericalLuneDetected;
            uint32_t triangleCount;
            float32_t solidAngles[5];
            float32_t totalSolidAngles;

            // Sampling ray visualization data
            uint32_t sampleCount;
            float32_t4 rayData[512]; // xyz = direction, w = PDF

            // Pyramid sampling debug data
            float32_t3 pyramidAxis1;         // First caliper axis direction
            float32_t3 pyramidAxis2;         // Second caliper axis direction
            float32_t3 pyramidCenter;        // Silhouette center direction
            float32_t pyramidHalfWidth1;     // Half-width along axis1 (sin-space)
            float32_t pyramidHalfWidth2;     // Half-width along axis2 (sin-space)
            float32_t pyramidOffset1;        // Center offset along axis1
            float32_t pyramidOffset2;        // Center offset along axis2
            float32_t pyramidSolidAngle;     // Bounding region solid angle
            uint32_t pyramidBestEdge;        // Which edge produced best caliper
            uint32_t pyramidSpansHemisphere; // Warning: silhouette >= hemisphere
            float32_t pyramidMin1;           // Min dot product along axis1
            float32_t pyramidMax1;           // Max dot product along axis1
            float32_t pyramidMin2;           // Min dot product along axis2
            float32_t pyramidMax2;           // Max dot product along axis2
            uint32_t axis2BiggerThanAxis1;

            // Sampling stats
            uint32_t validSampleCount;
            uint32_t threadCount; // Used as a hack for fragment shader, as dividend for validSampleCount
        };

#ifdef __HLSL_VERSION
        [[vk::binding(0, 0)]] RWStructuredBuffer<ResultData> DebugDataBuffer;
#endif

        struct PushConstants
        {
            float32_t3x4 modelMatrix;
            float32_t4 viewport;
            uint32_t sampleCount;
            uint32_t frameIndex;
        };

        struct PushConstantRayVis
        {
            float32_t4x4 viewProjMatrix;
            float32_t3x4 viewMatrix;
            float32_t3x4 modelMatrix;
            float32_t3x4 invModelMatrix;
            float32_t4 viewport;
            uint32_t frameIndex;
        };

        struct BenchmarkPushConstants
        {
            float32_t3x4 modelMatrix;
            uint32_t sampleCount;
        };

        static const float32_t3 colorLUT[27] = {
            float32_t3(0, 0, 0), float32_t3(0.5, 0.5, 0.5),
            float32_t3(1, 0, 0), float32_t3(0, 1, 0), float32_t3(0, 0, 1),
            float32_t3(1, 1, 0), float32_t3(1, 0, 1), float32_t3(0, 1, 1),
            float32_t3(1, 0.5, 0), float32_t3(1, 0.65, 0), float32_t3(0.8, 0.4, 0),
            float32_t3(1, 0.4, 0.7), float32_t3(1, 0.75, 0.8), float32_t3(0.7, 0.1, 0.3),
            float32_t3(0.5, 0, 0.5), float32_t3(0.6, 0.4, 0.8), float32_t3(0.3, 0, 0.5),
            float32_t3(0, 0.5, 0), float32_t3(0.5, 1, 0), float32_t3(0, 0.5, 0.25),
            float32_t3(0, 0, 0.5), float32_t3(0.3, 0.7, 1), float32_t3(0, 0.4, 0.6),
            float32_t3(0.6, 0.4, 0.2), float32_t3(0.8, 0.7, 0.3), float32_t3(0.4, 0.3, 0.1), float32_t3(1, 1, 1)};

#ifndef __HLSL_VERSION
        static const char *colorNames[27] = {"Black", "Gray", "Red", "Green", "Blue", "Yellow", "Magenta", "Cyan",
                                             "Orange", "Light Orange", "Dark Orange", "Pink", "Light Pink", "Deep Rose", "Purple", "Light Purple",
                                             "Indigo", "Dark Green", "Lime", "Forest Green", "Navy", "Sky Blue", "Teal", "Brown",
                                             "Tan/Beige", "Dark Brown", "White"};
#endif // __HLSL_VERSION
    }
}
#endif // _SOLID_ANGLE_VIS_EXAMPLE_COMMON_HLSL_INCLUDED_
