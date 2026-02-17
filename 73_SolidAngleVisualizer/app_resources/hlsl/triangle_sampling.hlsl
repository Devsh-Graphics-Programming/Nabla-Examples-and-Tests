//// Copyright (C) 2026-2026 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _SOLID_ANGLE_VIS_EXAMPLE_TRIANGLE_SAMPLING_HLSL_INCLUDED_
#define _SOLID_ANGLE_VIS_EXAMPLE_TRIANGLE_SAMPLING_HLSL_INCLUDED_

// Include the spherical triangle utilities
#include "gpu_common.hlsl"
#include <nbl/builtin/hlsl/shapes/spherical_triangle.hlsl>
#include <nbl/builtin/hlsl/sampling/spherical_triangle.hlsl>
#include <nbl/builtin/hlsl/sampling/projected_spherical_triangle.hlsl>
#include <nbl/builtin/hlsl/random/pcg.hlsl>
#include <nbl/builtin/hlsl/random/xoroshiro.hlsl>
#include "silhouette.hlsl"

using namespace nbl::hlsl;

// Maximum number of triangles we can have after clipping
// Without clipping, max 3 faces can be visible at once so 3 faces * 2 triangles = 6 edges, forming max 4 triangles
// With clipping, one more edge. 7 - 2 = 5 max triangles because fanning from one vertex
#define MAX_TRIANGLES 5

struct TriangleFanSampler
{
    uint32_t count;                               // Number of valid triangles
    uint32_t samplingMode;                        // Mode used during build
    float32_t totalWeight;                        // Sum of all triangle weights
    float32_t3 faceNormal;                        // Face normal (only used for projected mode)
    float32_t triangleSolidAngles[MAX_TRIANGLES]; // Weight per triangle (for selection)
    uint32_t triangleIndices[MAX_TRIANGLES];      // Vertex index i (forms triangle with v0, vi, vi+1)

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
        float32_t a = asin(clamp(l0, -1.0f, 1.0f)); // side v0-v1
        float32_t b = asin(clamp(l1, -1.0f, 1.0f)); // side v1-v2
        float32_t c = asin(clamp(l2, -1.0f, 1.0f)); // side v2-v0

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

    // Build fan triangulation, cache weights for triangle selection
    static TriangleFanSampler create(ClippedSilhouette silhouette, uint32_t mode)
    {
        TriangleFanSampler self;
        self.count = 0;
        self.totalWeight = 0.0f;
        self.samplingMode = mode;
        self.faceNormal = float32_t3(0, 0, 0);

        if (silhouette.count < 3)
            return self;

        const float32_t3 v0 = silhouette.vertices[0];
        const float32_t3 origin = float32_t3(0, 0, 0);

        // Compute face normal ONCE before the loop - silhouette is planar!
        if (mode == SAMPLING_MODE::TRIANGLE_PROJECTED_SOLID_ANGLE)
        {
            float32_t3 v1 = silhouette.vertices[1];
            float32_t3 v2 = silhouette.vertices[2];
            self.faceNormal = normalize(cross(v1 - v0, v2 - v0));
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
            if (mode == SAMPLING_MODE::TRIANGLE_PROJECTED_SOLID_ANGLE)
            {
                float32_t3 cos_vertices = clamp(
                    (shapeTri.cos_sides - shapeTri.cos_sides.yzx * shapeTri.cos_sides.zxy) *
                        shapeTri.csc_sides.yzx * shapeTri.csc_sides.zxy,
                    float32_t3(-1.0f, -1.0f, -1.0f),
                    float32_t3(1.0f, 1.0f, 1.0f));
                solidAngle = shapeTri.projectedSolidAngleOfTriangle(self.faceNormal, shapeTri.cos_sides, shapeTri.csc_sides, cos_vertices);
            }
            else
            {
                solidAngle = shapeTri.solidAngleOfTriangle();
            }

            if (solidAngle <= 0.0f)
                continue;

            // Store only what's needed for weighted selection
            self.triangleSolidAngles[self.count] = solidAngle;
            self.triangleIndices[self.count] = i;
            self.totalWeight += solidAngle;
            self.count++;
        }

#if DEBUG_DATA
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
        DebugDataBuffer[0].maxTrianglesExceeded = (self.count > MAX_TRIANGLES);
        DebugDataBuffer[0].triangleCount = self.count;
        DebugDataBuffer[0].totalSolidAngles = self.totalWeight;
        for (uint32_t tri = 0; tri < self.count; tri++)
        {
            DebugDataBuffer[0].solidAngles[tri] = self.triangleSolidAngles[tri];
        }
#endif

        return self;
    }

    // Sample using cached selection weights, recompute geometry on-demand
    float32_t3 sample(ClippedSilhouette silhouette, float32_t2 xi, out float32_t pdf, out uint32_t selectedIdx)
    {
        selectedIdx = 0;

        // Handle empty or invalid data
        if (count == 0 || totalWeight <= 0.0f)
        {
            pdf = 0.0f;
            return float32_t3(0, 0, 1);
        }

        // Select triangle using cached weighted random selection
        float32_t targetWeight = xi.x * totalWeight;
        float32_t cumulativeWeight = 0.0f;
        float32_t prevCumulativeWeight = 0.0f;

        NBL_UNROLL
        for (uint32_t i = 0; i < count; i++)
        {
            prevCumulativeWeight = cumulativeWeight;
            cumulativeWeight += triangleSolidAngles[i];

            if (targetWeight <= cumulativeWeight)
            {
                selectedIdx = i;
                break;
            }
        }

        // Remap xi.x to [0,1] within selected triangle's solidAngle interval
        float32_t triSolidAngle = triangleSolidAngles[selectedIdx];
        float32_t u = (targetWeight - prevCumulativeWeight) / max(triSolidAngle, 1e-7f);

        // Reconstruct the selected triangle geometry
        uint32_t vertexIdx = triangleIndices[selectedIdx];
        float32_t3 v0 = silhouette.vertices[0];
        float32_t3 v1 = silhouette.vertices[vertexIdx];
        float32_t3 v2 = silhouette.vertices[vertexIdx + 1];

        float32_t3 fn = normalize(cross(v1 - v0, v2 - v0));

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

        if (samplingMode == SAMPLING_MODE::TRIANGLE_PROJECTED_SOLID_ANGLE)
        {
            sampling::ProjectedSphericalTriangle<float32_t> samplingTri = sampling::ProjectedSphericalTriangle<float32_t>::create(shapeTri);

            direction = samplingTri.generate(rcpPdf, triSolidAngle, cos_vertices, sin_vertices, shapeTri.cos_sides[0], shapeTri.cos_sides[2], shapeTri.csc_sides[1], shapeTri.csc_sides[2], fn, false, float32_t2(u, xi.y));
            triSolidAngle = rcpPdf; // projected solid angle returned as rcpPdf
        }
        else
        {
            sampling::SphericalTriangle<float32_t> samplingTri = sampling::SphericalTriangle<float32_t>::create(shapeTri);
            direction = samplingTri.generate(triSolidAngle, cos_vertices, sin_vertices, shapeTri.cos_sides[0], shapeTri.cos_sides[2], shapeTri.csc_sides[1], shapeTri.csc_sides[2], float32_t2(u, xi.y));
        }

        // Calculate PDF
        float32_t trianglePdf = 1.0f / triSolidAngle;
        float32_t selectionProb = triSolidAngle / totalWeight;
        pdf = trianglePdf * selectionProb;

        return normalize(direction);
    }
};

#endif // _SOLID_ANGLE_VIS_EXAMPLE_TRIANGLE_SAMPLING_HLSL_INCLUDED_
