#pragma shader_stage(vertex)

#include "common.hlsl"
#include <nbl/builtin/hlsl/shapes/beziers.hlsl>
#include <nbl/builtin/hlsl/math/equations/quadratic.hlsl>
#include <nbl/builtin/hlsl/limits.hlsl>
#include <nbl/builtin/hlsl/algorithm.hlsl>

// TODO[Lucas]: Move these functions to builtin hlsl functions (Even the shadertoy obb and aabb ones)
float cross2D(float2 a, float2 b)
{
    return determinant(float2x2(a,b));
}

float2 BezierTangent(float2 p0, float2 p1, float2 p2, float t)
{
    return 2.0 * (1.0 - t) * (p1 - p0) + 2.0 * t * (p2 - p1);
}

float2 QuadraticBezier(float2 p0, float2 p1, float2 p2, float t)
{
    return shapes::QuadraticBezier<float>::construct(p0, p1, p2).evaluate(t);
}

struct NDCClipProjectionData
{
    pfloat64_t3x3 projectionToNDC; // pre-multiplied projection in a tree
    float32_t2 minClipNDC;
    float32_t2 maxClipNDC;
};

NDCClipProjectionData getClipProjectionData(in MainObject mainObj)
{
    NDCClipProjectionData ret;
    if (mainObj.customProjectionIndex != InvalidCustomProjectionIndex)
    {
        // If projection type is worldspace projection and clip:
        pfloat64_t3x3 customProjection = loadCustomProjection(mainObj.customProjectionIndex);
        ret.projectionToNDC = nbl::hlsl::mul(globals.defaultProjectionToNDC, customProjection);
    }
    else
        ret.projectionToNDC = globals.defaultProjectionToNDC;

    if (mainObj.customClipRectIndex != InvalidCustomClipRectIndex)
    {
        WorldClipRect worldClipRect = loadCustomClipRect(mainObj.customClipRectIndex);
        
        /// [NOTE]: Optimization: we avoid looking for min/max in the shader because minClip and maxClip in default worldspace are defined in such a way that minClip.y > maxClip.y so minClipNDC.y < maxClipNDC.y
        ret.minClipNDC = nbl::hlsl::_static_cast<float32_t2>(transformPointNdc(globals.defaultProjectionToNDC, worldClipRect.minClip));
        ret.maxClipNDC = nbl::hlsl::_static_cast<float32_t2>(transformPointNdc(globals.defaultProjectionToNDC, worldClipRect.maxClip));
    }
    else
    {
        ret.minClipNDC = float2(-1.0f, -1.0f);
        ret.maxClipNDC = float2(+1.0f, +1.0f);
    }
    
    if (mainObj.transformationType == TransformationType::TT_FIXED_SCREENSPACE_SIZE)
        ret.projectionToNDC = nbl::hlsl::mul(ret.projectionToNDC, globals.screenToWorldScaleTransform);
    
    return ret;
}

float2 transformPointScreenSpace(pfloat64_t3x3 transformation, uint32_t2 resolution, pfloat64_t2 point2d)
{
    pfloat64_t2 ndc = transformPointNdc(transformation, point2d);
    pfloat64_t2 result = (ndc + 1.0f) * 0.5f * _static_cast<pfloat64_t2>(resolution);

    return _static_cast<float2>(result);
}
float2 transformVectorScreenSpace(pfloat64_t3x3 transformation, uint32_t2 resolution, pfloat64_t2 vec2d)
{
     pfloat64_t2 ndc = transformVectorNdc(transformation, vec2d);
     pfloat64_t2 result = (ndc) * 0.5f * _static_cast<pfloat64_t2>(resolution);
     return _static_cast<float2>(result);
}
float32_t4 transformFromSreenSpaceToNdc(float2 pos, uint32_t2 resolution)
{
    return float32_t4((pos.xy / (float32_t2)resolution) * 2.0f - 1.0f, 0.0f, 1.0f);
}
float32_t getScreenToWorldRatio(pfloat64_t3x3 transformation, uint32_t2 resolution)
{
	pfloat64_t idx_0_0 = transformation[0u].x * (resolution.x / 2.0);
	pfloat64_t idx_1_0 = transformation[1u].x * (resolution.y / 2.0);
    float32_t2 firstCol; firstCol.x = _static_cast<float32_t>(idx_0_0); firstCol.y = _static_cast<float32_t>(idx_1_0); 
	return nbl::hlsl::length(firstCol); // TODO: Do length in fp64?
}

template<bool FragmentShaderPixelInterlock>
void dilateHatch(out float2 outOffsetVec, out float2 outUV, const float2 undilatedCorner, const float2 dilateRate, const float2 ndcAxisU, const float2 ndcAxisV);

// Dilate with ease, our transparency algorithm will handle the overlaps easily with the help of FragmentShaderPixelInterlock
template<>
void dilateHatch<true>(out float2 outOffsetVec, out float2 outUV, const float2 undilatedCorner, const float2 dilateRate, const float2 ndcAxisU, const float2 ndcAxisV)
{
    const float2 dilatationFactor = 1.0 + 2.0 * dilateRate;
    
    // cornerMultiplier stores the direction of the corner to dilate:
    // (-1,-1)|--|(1,-1)
    //        |  |
    // (-1,1) |--|(1,1)
    const float2 cornerMultiplier = float2(undilatedCorner * 2.0 - 1.0);
    outUV = float2((cornerMultiplier * dilatationFactor + 1.0) * 0.5);
    
    // vx/vy are vectors in direction of the box's axes and their length is equal to X pixels (X = globals.antiAliasingFactor + 1.0)
    // and we use them for dilation of X pixels in ndc space by adding them to the currentCorner in NDC space 
    const float2 vx = ndcAxisU * dilateRate.x;
    const float2 vy = ndcAxisV * dilateRate.y;
    outOffsetVec = vx * cornerMultiplier.x + vy * cornerMultiplier.y; // (0, 0) should do -vx-vy and (1, 1) should do +vx+vy
}

// Don't dilate which causes overlap of colors when no fragshaderInterlock which powers our transparency and overlap resolving algorithm
template<>
void dilateHatch<false>(out float2 outOffsetVec, out float2 outUV, const float2 undilatedCorner, const float2 dilateRate, const float2 ndcAxisU, const float2 ndcAxisV)
{
    outOffsetVec = float2(0.0f, 0.0f);
    outUV = undilatedCorner;
    // TODO: If it became a huge bummer on AMD devices we can consider dilating only in minor direction which may still avoid color overlaps
    // Or optionally we could dilate and stuff when we know this hatch is opaque (alpha = 1.0)
}

[shader("vertex")]
PSInput vtxMain(uint vertexID : SV_VertexID)
{
    NDCClipProjectionData clipProjectionData;
    
    PSInput outV;

    // Default Initialize PS Input
    outV.position.zw = float2(0.0, 1.0);
    outV.data1 = uint4(0, 0, 0, 0);
    outV.data2 = float4(0, 0, 0, 0);
    outV.data3 = float4(0, 0, 0, 0);
    outV.data4 = float4(0, 0, 0, 0);
    outV.interp_data5 = float4(0, 0, 0, 0);

    if (pc.isDTMRendering)
    {
        outV.setObjType(ObjectType::TRIANGLE_MESH);
        outV.setMainObjectIdx(pc.triangleMeshMainObjectIndex);
    
        TriangleMeshVertex vtx = vk::RawBufferLoad<TriangleMeshVertex>(pc.triangleMeshVerticesBaseAddress + sizeof(TriangleMeshVertex) * vertexID, 8u);

        MainObject mainObj = loadMainObject(pc.triangleMeshMainObjectIndex);
        clipProjectionData = getClipProjectionData(mainObj);

        float screenToWorldRatio = getScreenToWorldRatio(clipProjectionData.projectionToNDC, globals.resolution);
        float worldToScreenRatio = 1.0f / screenToWorldRatio;
        outV.setCurrentWorldToScreenRatio(worldToScreenRatio);
        
        // assuming there are 3 * N vertices, number of vertices is equal to number of indices and indices are sequential starting from 0
        float2 transformedOriginalPos;
        float2 transformedDilatedPos;
        {
            uint32_t currentVertexWithinTriangleIndex = vertexID % 3;
            uint32_t firstVertexOfCurrentTriangleIndex = vertexID - currentVertexWithinTriangleIndex;

            TriangleMeshVertex triangleVertices[3];
            triangleVertices[0] = vk::RawBufferLoad<TriangleMeshVertex>(pc.triangleMeshVerticesBaseAddress + sizeof(TriangleMeshVertex) * firstVertexOfCurrentTriangleIndex, 8u);
            triangleVertices[1] = vk::RawBufferLoad<TriangleMeshVertex>(pc.triangleMeshVerticesBaseAddress + sizeof(TriangleMeshVertex) * (firstVertexOfCurrentTriangleIndex + 1), 8u);
            triangleVertices[2] = vk::RawBufferLoad<TriangleMeshVertex>(pc.triangleMeshVerticesBaseAddress + sizeof(TriangleMeshVertex) * (firstVertexOfCurrentTriangleIndex + 2), 8u);
            transformedOriginalPos = transformPointScreenSpace(clipProjectionData.projectionToNDC, globals.resolution, triangleVertices[currentVertexWithinTriangleIndex].pos);

            pfloat64_t2 triangleCentroid;
            triangleCentroid.x = (triangleVertices[0].pos.x + triangleVertices[1].pos.x + triangleVertices[2].pos.x) / _static_cast<pfloat64_t>(3.0f);
            triangleCentroid.y = (triangleVertices[0].pos.y + triangleVertices[1].pos.y + triangleVertices[2].pos.y) / _static_cast<pfloat64_t>(3.0f);

            // move triangles to local space, with centroid at (0, 0)
            triangleVertices[0].pos = triangleVertices[0].pos - triangleCentroid;
            triangleVertices[1].pos = triangleVertices[1].pos - triangleCentroid;
            triangleVertices[2].pos = triangleVertices[2].pos - triangleCentroid;

            // TODO: calculate dialation factor
            // const float dilateByPixels = 0.5 * (dtmSettings.maxScreenSpaceLineWidth + dtmSettings.maxWorldSpaceLineWidth * screenToWorldRatio) + aaFactor;
        
            pfloat64_t dialationFactor = _static_cast<pfloat64_t>(2.0f);
            pfloat64_t2 dialatedVertex = triangleVertices[currentVertexWithinTriangleIndex].pos * dialationFactor;

            dialatedVertex = dialatedVertex + triangleCentroid;

            transformedDilatedPos = transformPointScreenSpace(clipProjectionData.projectionToNDC, globals.resolution, dialatedVertex);
        }

        outV.position = transformFromSreenSpaceToNdc(transformedDilatedPos, globals.resolution);
        const float heightAsFloat = nbl::hlsl::_static_cast<float>(vtx.height);
        outV.setScreenSpaceVertexAttribs(float3(transformedOriginalPos, heightAsFloat));

        // full screen triangle (this will destroy outline, contour line and height drawing)
#if 0
        const uint vertexIdx = vertexID % 3;
        if(vertexIdx == 0)
            outV.position.xy = float2(-1.0f, -1.0f);
        else if (vertexIdx == 1)
            outV.position.xy = float2(-1.0f, 3.0f);
        else if (vertexIdx == 2)
            outV.position.xy = float2(3.0f, -1.0f);
#endif
    }
    else
    {
        const uint vertexIdx = vertexID & 0x3u;
        const uint objectID = vertexID >> 2;

        DrawObject drawObj = loadDrawObject(objectID);

        ObjectType objType = (ObjectType)(drawObj.type_subsectionIdx & 0x0000FFFF);
        uint32_t subsectionIdx = drawObj.type_subsectionIdx >> 16;
        outV.setObjType(objType);
        outV.setMainObjectIdx(drawObj.mainObjIndex);

        MainObject mainObj = loadMainObject(drawObj.mainObjIndex);
        clipProjectionData = getClipProjectionData(mainObj);
        
        float screenToWorldRatio = getScreenToWorldRatio(clipProjectionData.projectionToNDC, globals.resolution);
        float worldToScreenRatio = 1.0f / screenToWorldRatio;
        outV.setCurrentWorldToScreenRatio(worldToScreenRatio);
    
        // We only need these for Outline type objects like lines and bezier curves
        if (objType == ObjectType::LINE || objType == ObjectType::QUAD_BEZIER || objType == ObjectType::POLYLINE_CONNECTOR)
        {
            LineStyle lineStyle = loadLineStyle(mainObj.styleIdx);

            // Width is on both sides, thickness is one one side of the curve (div by 2.0f)
            const float screenSpaceLineWidth = lineStyle.screenSpaceLineWidth + lineStyle.worldSpaceLineWidth * screenToWorldRatio;
            const float antiAliasedLineThickness = screenSpaceLineWidth * 0.5f + globals.antiAliasingFactor;
            const float sdfLineThickness = screenSpaceLineWidth / 2.0f;
            outV.setLineThickness(sdfLineThickness);

            if (objType == ObjectType::LINE)
            {
                pfloat64_t2 points[2u];
                points[0u] = vk::RawBufferLoad<pfloat64_t2>(globals.pointers.geometryBuffer + drawObj.geometryAddress, 8u);
                points[1u] = vk::RawBufferLoad<pfloat64_t2>(globals.pointers.geometryBuffer + drawObj.geometryAddress + sizeof(LinePointInfo), 8u);

                const float phaseShift = vk::RawBufferLoad<float>(globals.pointers.geometryBuffer + drawObj.geometryAddress + sizeof(pfloat64_t2), 8u);
                const float patternStretch = vk::RawBufferLoad<float>(globals.pointers.geometryBuffer + drawObj.geometryAddress + sizeof(pfloat64_t2) + sizeof(float), 8u);
                outV.setCurrentPhaseShift(phaseShift);
                outV.setPatternStretch(patternStretch);

                float2 transformedPoints[2u];
                for (uint i = 0u; i < 2u; ++i)
                {
                    transformedPoints[i] = transformPointScreenSpace(clipProjectionData.projectionToNDC, globals.resolution, points[i]);
                }

                const float2 lineVector = normalize(transformedPoints[1u] - transformedPoints[0u]);
                const float2 normalToLine = float2(-lineVector.y, lineVector.x);

                if (vertexIdx == 0u || vertexIdx == 1u)
                {
                    // work in screen space coordinates because of fixed pixel size
                    outV.position.xy = transformedPoints[0u]
                        + normalToLine * (((float)vertexIdx - 0.5f) * 2.0f * antiAliasedLineThickness)
                        - lineVector * antiAliasedLineThickness;
                }
                else // if (vertexIdx == 2u || vertexIdx == 3u)
                {
                    // work in screen space coordinates because of fixed pixel size
                    outV.position.xy = transformedPoints[1u]
                        + normalToLine * (((float)vertexIdx - 2.5f) * 2.0f * antiAliasedLineThickness)
                        + lineVector * antiAliasedLineThickness;
                }

                outV.setLineStart(transformedPoints[0u]);
                outV.setLineEnd(transformedPoints[1u]);

                outV.position.xy = transformFromSreenSpaceToNdc(outV.position.xy, globals.resolution).xy;
            }
            else if (objType == ObjectType::QUAD_BEZIER)
            {
                pfloat64_t2 points[3u];
                points[0u] = vk::RawBufferLoad<pfloat64_t2>(globals.pointers.geometryBuffer + drawObj.geometryAddress, 8u);
                points[1u] = vk::RawBufferLoad<pfloat64_t2>(globals.pointers.geometryBuffer + drawObj.geometryAddress + sizeof(pfloat64_t2), 8u);
                points[2u] = vk::RawBufferLoad<pfloat64_t2>(globals.pointers.geometryBuffer + drawObj.geometryAddress + sizeof(pfloat64_t2) * 2u, 8u);

                const float phaseShift = vk::RawBufferLoad<float>(globals.pointers.geometryBuffer + drawObj.geometryAddress + sizeof(pfloat64_t2) * 3u, 8u);
                const float patternStretch = vk::RawBufferLoad<float>(globals.pointers.geometryBuffer + drawObj.geometryAddress + sizeof(pfloat64_t2) * 3u + sizeof(float), 8u);
                outV.setCurrentPhaseShift(phaseShift);
                outV.setPatternStretch(patternStretch);

                // transform these points into screen space and pass to fragment
                float2 transformedPoints[3u];
                for (uint i = 0u; i < 3u; ++i)
                {
                    transformedPoints[i] = transformPointScreenSpace(clipProjectionData.projectionToNDC, globals.resolution, points[i]);
                }

                shapes::QuadraticBezier<float> quadraticBezier = shapes::QuadraticBezier<float>::construct(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u]);
                shapes::Quadratic<float> quadratic = shapes::Quadratic<float>::constructFromBezier(quadraticBezier);
                shapes::Quadratic<float>::ArcLengthCalculator preCompData = shapes::Quadratic<float>::ArcLengthCalculator::construct(quadratic);

                outV.setQuadratic(quadratic);
                outV.setQuadraticPrecomputedArcLenData(preCompData);

                float2 Mid = (transformedPoints[0u] + transformedPoints[2u]) / 2.0f;
                float Radius = length(Mid - transformedPoints[0u]) / 2.0f;

                // https://algorithmist.wordpress.com/2010/12/01/quad-bezier-curvature/
                float2 vectorAB = transformedPoints[1u] - transformedPoints[0u];
                float2 vectorAC = transformedPoints[2u] - transformedPoints[1u];
                float area = abs(vectorAB.x * vectorAC.y - vectorAB.y * vectorAC.x) * 0.5;
                float MaxCurvature;
                if (length(transformedPoints[1u] - lerp(transformedPoints[0u], transformedPoints[2u], 0.25f)) > Radius && length(transformedPoints[1u] - lerp(transformedPoints[0u], transformedPoints[2u], 0.75f)) > Radius)
                    MaxCurvature = pow(length(transformedPoints[1u] - Mid), 3) / (area * area);
                else
                    MaxCurvature = max(area / pow(length(transformedPoints[0u] - transformedPoints[1u]), 3), area / pow(length(transformedPoints[2u] - transformedPoints[1u]), 3));

                // We only do this adaptive thing when "MinRadiusOfOsculatingCircle = RadiusOfMaxCurvature < screenSpaceLineWidth/4" OR "MaxCurvature > 4/screenSpaceLineWidth";
                //  which means there is a self intersection because of large lineWidth relative to the curvature (in screenspace)
                //  the reason for division by 4.0f is 1. screenSpaceLineWidth is expanded on both sides and 2. the fact that diameter/2=radius, 
                const bool noCurvature = abs(dot(normalize(vectorAB), normalize(vectorAC)) - 1.0f) < exp2(-10.0f);
                if (MaxCurvature * screenSpaceLineWidth > 4.0f || noCurvature)
                {
                    //OBB Fallback
                    float2 obbV0;
                    float2 obbV1;
                    float2 obbV2;
                    float2 obbV3;
                    quadraticBezier.computeOBB(antiAliasedLineThickness, obbV0, obbV1, obbV2, obbV3);
                    if (subsectionIdx == 0)
                    {
                        if (vertexIdx == 0u)
                            outV.position = float4(obbV0, 0.0, 1.0f);
                        else if (vertexIdx == 1u)
                            outV.position = float4(obbV1, 0.0, 1.0f);
                        else if (vertexIdx == 2u)
                            outV.position = float4(obbV3, 0.0, 1.0f);
                        else if (vertexIdx == 3u)
                            outV.position = float4(obbV2, 0.0, 1.0f);
                    }
                    else
                        outV.position = float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
                else
                {
                    // this optimal value is hardcoded based on tests and benchmarks of pixel shader invocation
                    // this is the place where we use it's tangent in the bezier to form sides the cages
                    const float optimalT = 0.145f;

                    // Whether or not to flip the the interior cage nodes
                    int flip = cross2D(transformedPoints[0u] - transformedPoints[1u], transformedPoints[2u] - transformedPoints[1u]) > 0.0f ? -1 : 1;

                    const float middleT = 0.5f;
                    float2 midPos = QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], middleT);
                    float2 midTangent = normalize(BezierTangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], middleT));
                    float2 midNormal = float2(-midTangent.y, midTangent.x) * flip;

                    /*
                                P1
                                +


                   exterior0              exterior1
                      ----------------------
                     /                      \-
                   -/    ----------------     \
                  /    -/interior0     interior1
                 /    /                    \    \-
               -/   -/                      \-    \
              /   -/                          \    \-
             /   /                             \-    \
         P0 +                                    \    + P2
                    */

                    // Internal cage points
                    float2 interior0;
                    float2 interior1;

                    float2 middleExteriorPoint = midPos - midNormal * antiAliasedLineThickness;


                    float2 leftTangent = normalize(BezierTangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], optimalT));
                    float2 leftNormal = normalize(float2(-leftTangent.y, leftTangent.x)) * flip;
                    float2 leftExteriorPoint = QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], optimalT) - leftNormal * antiAliasedLineThickness;
                    float2 exterior0 = shapes::util::LineLineIntersection<float>(middleExteriorPoint, midTangent, leftExteriorPoint, leftTangent);

                    float2 rightTangent = normalize(BezierTangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 1.0f - optimalT));
                    float2 rightNormal = normalize(float2(-rightTangent.y, rightTangent.x)) * flip;
                    float2 rightExteriorPoint = QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 1.0f - optimalT) - rightNormal * antiAliasedLineThickness;
                    float2 exterior1 = shapes::util::LineLineIntersection<float>(middleExteriorPoint, midTangent, rightExteriorPoint, rightTangent);

                    // Interiors
                    {
                        float2 tangent = normalize(BezierTangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.286f));
                        float2 normal = normalize(float2(-tangent.y, tangent.x)) * flip;
                        interior0 = QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.286) + normal * antiAliasedLineThickness;
                    }
                    {
                        float2 tangent = normalize(BezierTangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.714f));
                        float2 normal = normalize(float2(-tangent.y, tangent.x)) * flip;
                        interior1 = QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.714f) + normal * antiAliasedLineThickness;
                    }

                    if (subsectionIdx == 0u)
                    {
                        float2 endPointTangent = normalize(transformedPoints[1u] - transformedPoints[0u]);
                        float2 endPointNormal = float2(-endPointTangent.y, endPointTangent.x) * flip;
                        float2 endPointExterior = transformedPoints[0u] - endPointTangent * antiAliasedLineThickness;

                        if (vertexIdx == 0u)
                            outV.position = float4(shapes::util::LineLineIntersection<float>(leftExteriorPoint, leftTangent, endPointExterior, endPointNormal), 0.0, 1.0f);
                        else if (vertexIdx == 1u)
                            outV.position = float4(transformedPoints[0u] + endPointNormal * antiAliasedLineThickness - endPointTangent * antiAliasedLineThickness, 0.0, 1.0f);
                        else if (vertexIdx == 2u)
                            outV.position = float4(exterior0, 0.0, 1.0f);
                        else if (vertexIdx == 3u)
                            outV.position = float4(interior0, 0.0, 1.0f);
                    }
                    else if (subsectionIdx == 1u)
                    {
                        if (vertexIdx == 0u)
                            outV.position = float4(exterior0, 0.0, 1.0f);
                        else if (vertexIdx == 1u)
                            outV.position = float4(interior0, 0.0, 1.0f);
                        else if (vertexIdx == 2u)
                            outV.position = float4(exterior1, 0.0, 1.0f);
                        else if (vertexIdx == 3u)
                            outV.position = float4(interior1, 0.0, 1.0f);
                    }
                    else if (subsectionIdx == 2u)
                    {
                        float2 endPointTangent = normalize(transformedPoints[2u] - transformedPoints[1u]);
                        float2 endPointNormal = float2(-endPointTangent.y, endPointTangent.x) * flip;
                        float2 endPointExterior = transformedPoints[2u] + endPointTangent * antiAliasedLineThickness;

                        if (vertexIdx == 0u)
                            outV.position = float4(shapes::util::LineLineIntersection<float>(rightExteriorPoint, rightTangent, endPointExterior, endPointNormal), 0.0, 1.0f);
                        else if (vertexIdx == 1u)
                            outV.position = float4(transformedPoints[2u] + endPointNormal * antiAliasedLineThickness + endPointTangent * antiAliasedLineThickness, 0.0, 1.0f);
                        else if (vertexIdx == 2u)
                            outV.position = float4(exterior1, 0.0, 1.0f);
                        else if (vertexIdx == 3u)
                            outV.position = float4(interior1, 0.0, 1.0f);
                    }
                }

                outV.position.xy = (outV.position.xy / globals.resolution) * 2.0f - 1.0f;
            }
            else if (objType == ObjectType::POLYLINE_CONNECTOR)
            {
                const float FLOAT_INF = numeric_limits<float>::infinity;
                const float4 INVALID_VERTEX = float4(FLOAT_INF, FLOAT_INF, FLOAT_INF, FLOAT_INF);

                if (lineStyle.isRoadStyleFlag)
                {
                    const pfloat64_t2 circleCenter = vk::RawBufferLoad<pfloat64_t2>(globals.pointers.geometryBuffer + drawObj.geometryAddress, 8u);
                    const float2 v = vk::RawBufferLoad<float2>(globals.pointers.geometryBuffer + drawObj.geometryAddress + sizeof(pfloat64_t2), 8u);
                    const float cosHalfAngleBetweenNormals = vk::RawBufferLoad<float>(globals.pointers.geometryBuffer + drawObj.geometryAddress + sizeof(pfloat64_t2) + sizeof(float2), 8u);

                    const float2 circleCenterScreenSpace = transformPointScreenSpace(clipProjectionData.projectionToNDC, globals.resolution, circleCenter);
                    outV.setPolylineConnectorCircleCenter(circleCenterScreenSpace);

                    // to better understand variables at play, and the circle space, see documentation of `miterSDF` in fragment shader
                    // length of vector from circle center to intersection position (normalized so that circle radius = line thickness = 1.0)
                    float vLen = length(v);
                    float2 intersectionDirection_Screenspace = normalize(transformVectorScreenSpace(clipProjectionData.projectionToNDC, globals.resolution, _static_cast<pfloat64_t2>(v)));
                    const float2 v_Screenspace = intersectionDirection_Screenspace * vLen;

                    // Find other miter vertices
                    const float sinHalfAngleBetweenNormals = sqrt(1.0f - (cosHalfAngleBetweenNormals * cosHalfAngleBetweenNormals));
                    const float32_t2x2 rotationMatrix = float32_t2x2(cosHalfAngleBetweenNormals, -sinHalfAngleBetweenNormals, sinHalfAngleBetweenNormals, cosHalfAngleBetweenNormals);

                    // Pass the precomputed trapezoid values for the sdf
                    {
                        float longBase = sinHalfAngleBetweenNormals;
                        float shortBase = max((vLen - globals.miterLimit) * cosHalfAngleBetweenNormals / sinHalfAngleBetweenNormals, 0.0);
                        // height of the trapezoid / triangle
                        float hLen = min(globals.miterLimit, vLen);

                        outV.setPolylineConnectorTrapezoidStart(-1.0 * intersectionDirection_Screenspace * sdfLineThickness);
                        outV.setPolylineConnectorTrapezoidEnd(intersectionDirection_Screenspace * hLen * sdfLineThickness);
                        outV.setPolylineConnectorTrapezoidLongBase(sinHalfAngleBetweenNormals * ((1.0 + vLen) / (vLen - cosHalfAngleBetweenNormals)) * sdfLineThickness);
                        outV.setPolylineConnectorTrapezoidShortBase(shortBase * sdfLineThickness);
                    }

                    if (vertexIdx == 0u)
                    {
                        // multiplying the other way to rotate by -theta
                        const float2 V1 = normalize(mul(v_Screenspace, rotationMatrix)) * antiAliasedLineThickness * 2.0f;
                        const float2 screenSpaceV1 = circleCenterScreenSpace + V1;
                        outV.position = float4(screenSpaceV1, 0.0f, 1.0f);   
                    }
                    else if (vertexIdx == 1u)
                    {
                        outV.position = float4(circleCenterScreenSpace, 0.0f, 1.0f);
                    }
                    else if (vertexIdx == 2u)
                    {
                        // find intersection point vertex
                        float2 intersectionPoint = v_Screenspace * antiAliasedLineThickness * 2.0f;
                        intersectionPoint += circleCenterScreenSpace;
                        outV.position = float4(intersectionPoint, 0.0f, 1.0f);
                    }
                    else if (vertexIdx == 3u)
                    {
                        const float2 V2 = normalize(mul(rotationMatrix, v_Screenspace)) * antiAliasedLineThickness * 2.0f;
                        const float2 screenSpaceV2 = circleCenterScreenSpace + V2;
                        outV.position = float4(screenSpaceV2, 0.0f, 1.0f);
                    }

                    outV.position.xy = transformFromSreenSpaceToNdc(outV.position.xy, globals.resolution).xy;
                }
                else
                {
                    outV.position = INVALID_VERTEX;
                }
            }
        }
        else if (objType == ObjectType::CURVE_BOX)
        {
            CurveBox curveBox;
            curveBox.aabbMin = vk::RawBufferLoad<pfloat64_t2>(globals.pointers.geometryBuffer + drawObj.geometryAddress, 8u);
            curveBox.aabbMax = vk::RawBufferLoad<pfloat64_t2>(globals.pointers.geometryBuffer + drawObj.geometryAddress + sizeof(pfloat64_t2), 8u);

            for (uint32_t i = 0; i < 3; i ++)
            {
                curveBox.curveMin[i] = vk::RawBufferLoad<float32_t2>(globals.pointers.geometryBuffer + drawObj.geometryAddress + sizeof(pfloat64_t2) * 2 + sizeof(float32_t2) * i, 4u);
                curveBox.curveMax[i] = vk::RawBufferLoad<float32_t2>(globals.pointers.geometryBuffer + drawObj.geometryAddress + sizeof(pfloat64_t2) * 2 + sizeof(float32_t2) * (3 + i), 4u);
            }

            pfloat64_t2 aabbMaxXMinY;
            aabbMaxXMinY.x = curveBox.aabbMax.x;
            aabbMaxXMinY.y = curveBox.aabbMin.y;

            pfloat64_t2 aabbMinXMaxY;
            aabbMinXMaxY.x = curveBox.aabbMin.x;
            aabbMinXMaxY.y = curveBox.aabbMax.y;

            const float2 ndcAxisU = _static_cast<float2>(transformVectorNdc(clipProjectionData.projectionToNDC, aabbMaxXMinY - curveBox.aabbMin));
            const float2 ndcAxisV = _static_cast<float2>(transformVectorNdc(clipProjectionData.projectionToNDC, aabbMinXMaxY - curveBox.aabbMin));

            const float2 screenSpaceAabbExtents = float2(length(ndcAxisU * float2(globals.resolution)) / 2.0, length(ndcAxisV * float2(globals.resolution)) / 2.0);

            // we could use something like  this to compute screen space change over minor/major change and avoid ddx(minor), ddy(major) in frag shader (the code below doesn't account for rotation)
            outV.setCurveBoxScreenSpaceSize(float2(screenSpaceAabbExtents));
        
            const float2 undilatedCorner = float2(bool2(vertexIdx & 0x1u, vertexIdx >> 1));
            const pfloat64_t2 undilatedCornerF64 = _static_cast<pfloat64_t2>(undilatedCorner);

            // We don't dilate on AMD (= no fragShaderInterlock)
            const float pixelsToIncreaseOnEachSide = globals.antiAliasingFactor + 1.0;
            const float2 dilateRate = pixelsToIncreaseOnEachSide / screenSpaceAabbExtents; // float sufficient to hold the dilate rect? 
            float2 dilateVec;
            float2 dilatedUV;
            dilateHatch<DeviceConfigCaps::fragmentShaderPixelInterlock>(dilateVec, dilatedUV, undilatedCorner, dilateRate, ndcAxisU, ndcAxisV);

            // doing interpolation this way to ensure correct endpoints and 0 and 1, we can alternatively use branches to set current corner based on vertexIdx
            const pfloat64_t2 currentCorner = curveBox.aabbMin * (_static_cast<pfloat64_t2>(float2(1.0f, 1.0f)) - undilatedCornerF64) +
                curveBox.aabbMax * undilatedCornerF64;

            const float2 coord = _static_cast<float2>(transformPointNdc(clipProjectionData.projectionToNDC, currentCorner) + _static_cast<pfloat64_t2>(dilateVec));

            outV.position = float4(coord, 0.f, 1.f);
 
            const uint major = (uint)SelectedMajorAxis;
            const uint minor = 1-major;

            // A, B & C get converted from unorm to [0, 1]
            // A & B get converted from [0,1] to [-2, 2]
            shapes::Quadratic<float> curveMin = shapes::Quadratic<float>::construct(
                curveBox.curveMin[0], curveBox.curveMin[1], curveBox.curveMin[2]);
            shapes::Quadratic<float> curveMax = shapes::Quadratic<float>::construct(
                curveBox.curveMax[0], curveBox.curveMax[1], curveBox.curveMax[2]);

            outV.setMinorBBoxUV(dilatedUV[minor]);
            outV.setMajorBBoxUV(dilatedUV[major]);

            outV.setCurveMinMinor(math::equations::Quadratic<float>::construct(
                curveMin.A[minor], 
                curveMin.B[minor], 
                curveMin.C[minor]));
            outV.setCurveMinMajor(math::equations::Quadratic<float>::construct(
                curveMin.A[major], 
                curveMin.B[major], 
                curveMin.C[major]));

            outV.setCurveMaxMinor(math::equations::Quadratic<float>::construct(
                curveMax.A[minor], 
                curveMax.B[minor], 
                curveMax.C[minor]));
            outV.setCurveMaxMajor(math::equations::Quadratic<float>::construct(
                curveMax.A[major], 
                curveMax.B[major], 
                curveMax.C[major]));

            //math::equations::Quadratic<float> curveMinRootFinding = math::equations::Quadratic<float>::construct(
            //    curveMin.A[major], 
            //    curveMin.B[major], 
            //    curveMin.C[major] - maxCorner[major]);
            //math::equations::Quadratic<float> curveMaxRootFinding = math::equations::Quadratic<float>::construct(
            //    curveMax.A[major], 
            //    curveMax.B[major], 
            //    curveMax.C[major] - maxCorner[major]);
            //outV.setMinCurvePrecomputedRootFinders(PrecomputedRootFinder<float>::construct(curveMinRootFinding));
            //outV.setMaxCurvePrecomputedRootFinders(PrecomputedRootFinder<float>::construct(curveMaxRootFinding));
        }
        else if (objType == ObjectType::FONT_GLYPH)
        {
            LineStyle lineStyle = loadLineStyle(mainObj.styleIdx);
            const float italicTiltSlope = lineStyle.screenSpaceLineWidth; // aliased text style member with line style
        
            GlyphInfo glyphInfo;
            glyphInfo.topLeft = vk::RawBufferLoad<pfloat64_t2>(globals.pointers.geometryBuffer + drawObj.geometryAddress, 8u);
            glyphInfo.dirU = vk::RawBufferLoad<float32_t2>(globals.pointers.geometryBuffer + drawObj.geometryAddress + sizeof(pfloat64_t2), 4u);
            glyphInfo.aspectRatio = vk::RawBufferLoad<float32_t>(globals.pointers.geometryBuffer + drawObj.geometryAddress + sizeof(pfloat64_t2) + sizeof(float2), 4u);
            glyphInfo.minUV_textureID_packed = vk::RawBufferLoad<uint32_t>(globals.pointers.geometryBuffer + drawObj.geometryAddress + sizeof(pfloat64_t2) + sizeof(float2) + sizeof(float), 4u);

            float32_t2 minUV = glyphInfo.getMinUV();
            uint16_t textureID = glyphInfo.getTextureID();

            const float32_t2 dirV = float32_t2(glyphInfo.dirU.y, -glyphInfo.dirU.x) * glyphInfo.aspectRatio;
            const float2 screenTopLeft = _static_cast<float2>(transformPointNdc(clipProjectionData.projectionToNDC, glyphInfo.topLeft));
            const float2 screenDirU = _static_cast<float2>(transformVectorNdc(clipProjectionData.projectionToNDC, _static_cast<pfloat64_t2>(glyphInfo.dirU)));
            const float2 screenDirV = _static_cast<float2>(transformVectorNdc(clipProjectionData.projectionToNDC, _static_cast<pfloat64_t2>(dirV)));

            const float2 corner = float2(bool2(vertexIdx & 0x1u, vertexIdx >> 1)); // corners of square from (0, 0) to (1, 1)
            const float2 undilatedCornerNDC = corner * 2.0 - 1.0; // corners of square from (-1, -1) to (1, 1)
        
            const float2 screenSpaceAabbExtents = float2(length(screenDirU * float2(globals.resolution)) / 2.0, length(screenDirV * float2(globals.resolution)) / 2.0);
            const float pixelsToIncreaseOnEachSide = globals.antiAliasingFactor + 1.0;
            const float2 dilateRate = (pixelsToIncreaseOnEachSide / screenSpaceAabbExtents);

            const float2 vx = screenDirU * dilateRate.x;
            const float2 vy = screenDirV * dilateRate.y;
            const float2 offsetVec = vx * undilatedCornerNDC.x + vy * undilatedCornerNDC.y;
            float2 coord = screenTopLeft + corner.x * screenDirU + corner.y * screenDirV + offsetVec;

            if (corner.y == 0 && italicTiltSlope > 0.0f)
                coord += normalize(screenDirU) * length(screenDirV) * italicTiltSlope * float(globals.resolution.y) / float(globals.resolution.x);
        
            // If aspect ratio of the dimensions and glyph inside the texture are the same then screenPxRangeX === screenPxRangeY
            // but if the glyph box is stretched in any way then we won't get correct msdf
            // in that case we need to take the max(screenPxRangeX, screenPxRangeY) to avoid blur due to underexaggerated distances
            // We compute screenPxRange using the ratio of our screenspace extent to the texel space our glyph takes inside the texture
            // Our glyph is centered inside the texture, so `maxUV = 1.0 - minUV` and `glyphTexelSize = (1.0-2.0*minUV) * MSDFSize
            const float screenPxRangeX = screenSpaceAabbExtents.x / ((1.0 - 2.0 * minUV.x)); // division by MSDFSize happens after max
            const float screenPxRangeY = screenSpaceAabbExtents.y / ((1.0 - 2.0 * minUV.y)); // division by MSDFSize happens after max
            outV.setFontGlyphPxRange((max(max(screenPxRangeX, screenPxRangeY), 1.0) * MSDFPixelRangeHalf) / MSDFSize); // we premultuply by MSDFPixelRange/2.0, to avoid doing it in frag shader

            // In order to keep the shape scale constant with any dilation values:
            // We compute the new dilated minUV that gets us minUV when interpolated on the previous undilated top left
            const float2 topLeftInterpolationValue = (dilateRate/(1.0+2.0*dilateRate));
            const float2 dilatedMinUV = (topLeftInterpolationValue - minUV) / (2.0 * topLeftInterpolationValue - 1.0);
            const float2 dilatedMaxUV = float2(1.0, 1.0) - dilatedMinUV;
        
            const float2 uv = dilatedMinUV + corner * (dilatedMaxUV - dilatedMinUV);

            outV.position = float4(coord, 0.f, 1.f);
            outV.setFontGlyphUV(uv);
            outV.setFontGlyphTextureId(textureID);
        }
        else if (objType == ObjectType::STATIC_IMAGE)
        {
            pfloat64_t2 topLeft = vk::RawBufferLoad<pfloat64_t2>(globals.pointers.geometryBuffer + drawObj.geometryAddress, 8u);
            float32_t2 dirU = vk::RawBufferLoad<float32_t2>(globals.pointers.geometryBuffer + drawObj.geometryAddress + sizeof(pfloat64_t2), 4u);
            float32_t aspectRatio = vk::RawBufferLoad<float32_t>(globals.pointers.geometryBuffer + drawObj.geometryAddress + sizeof(pfloat64_t2) + sizeof(float2), 4u);
            uint32_t textureID = vk::RawBufferLoad<uint32_t>(globals.pointers.geometryBuffer + drawObj.geometryAddress + sizeof(pfloat64_t2) + sizeof(float2) + sizeof(float), 4u);

            // TODO[DEVSH]: make sure it's documented properly that for topLeft+dirV+aspectRatio to work it's computing dirU like below (they need to be careful with transformations when y increases when you go down in screen
            const float32_t2 dirV = float32_t2(dirU.y, -dirU.x) * aspectRatio;
            const float2 ndcTopLeft = _static_cast<float2>(transformPointNdc(clipProjectionData.projectionToNDC, topLeft));
            const float2 ndcDirU = _static_cast<float2>(transformVectorNdc(clipProjectionData.projectionToNDC, _static_cast<pfloat64_t2>(dirU)));
            const float2 ndcDirV = _static_cast<float2>(transformVectorNdc(clipProjectionData.projectionToNDC, _static_cast<pfloat64_t2>(dirV)));

            float2 corner = float2(bool2(vertexIdx & 0x1u, vertexIdx >> 1));
            float2 uv = corner; // non-dilated
        
            float2 ndcCorner = ndcTopLeft + corner.x * ndcDirU + corner.y * ndcDirV;
        
            outV.position = float4(ndcCorner, 0.f, 1.f);
            outV.setImageUV(uv);
            outV.setImageTextureId(textureID);
        }
        else if (objType == ObjectType::GRID_DTM)
        {
            pfloat64_t2 topLeft = vk::RawBufferLoad<pfloat64_t2>(globals.pointers.geometryBuffer + drawObj.geometryAddress, 8u);
            const pfloat64_t2 worldSpaceExtents = vk::RawBufferLoad<pfloat64_t2>(globals.pointers.geometryBuffer + drawObj.geometryAddress + sizeof(pfloat64_t2), 8u);
            uint32_t textureID = vk::RawBufferLoad<uint32_t>(globals.pointers.geometryBuffer + drawObj.geometryAddress + 2 * sizeof(pfloat64_t2), 8u);
            float gridCellWidth = vk::RawBufferLoad<float>(globals.pointers.geometryBuffer + drawObj.geometryAddress + 2 * sizeof(pfloat64_t2) + sizeof(uint32_t), 8u);
            float thicknessOfTheThickestLine = vk::RawBufferLoad<float>(globals.pointers.geometryBuffer + drawObj.geometryAddress + 2 * sizeof(pfloat64_t2) + sizeof(uint32_t) + sizeof(float), 8u);

            // TODO: remove
            // test large dilation
            //thicknessOfTheThickestLine += 200.0f;

            const float2 corner = float2(bool2(vertexIdx & 0x1u, vertexIdx >> 1));

            outV.setGridDTMHeightTextureID(textureID);
            outV.setGridDTMScreenSpaceCellWidth(gridCellWidth * screenToWorldRatio);
            outV.setGridDTMScreenSpaceGridExtents(_static_cast<float2>(worldSpaceExtents) * screenToWorldRatio);

            static const float SquareRootOfTwo = 1.4142135f;
            const pfloat64_t dilationFactor = _static_cast<pfloat64_t>(SquareRootOfTwo * thicknessOfTheThickestLine);
            pfloat64_t2 dilationVector;
            dilationVector.x = dilationFactor;
            dilationVector.y = dilationFactor;

            const pfloat64_t dilationFactorTimesTwo = dilationFactor * 2.0f;
            pfloat64_t2 dilationFactorTimesTwoVector;
            dilationFactorTimesTwoVector.x = dilationFactorTimesTwo;
            dilationFactorTimesTwoVector.y = dilationFactorTimesTwo;
            const pfloat64_t2 dilatedGridExtents = worldSpaceExtents + dilationFactorTimesTwoVector;
            const float2 uvScale = _static_cast<float2>(worldSpaceExtents) / _static_cast<float2>(dilatedGridExtents);
            float2 uvOffset = _static_cast<float2>(dilationVector) / _static_cast<float2>(dilatedGridExtents);
            uvOffset /= uvScale;

            if (corner.x == 0.0f && corner.y == 0.0f)
            {
                dilationVector.x = ieee754::flipSign(dilationVector.x, true);
                uvOffset.x = -uvOffset.x;
                uvOffset.y = -uvOffset.y;
            }
            else if (corner.x == 0.0f && corner.y == 1.0f)
            {
                dilationVector.x = ieee754::flipSign(dilationVector.x, true);
                dilationVector.y = ieee754::flipSign(dilationVector.y, true);
                uvOffset.x = -uvOffset.x;
            }
            else if (corner.x == 1.0f && corner.y == 1.0f)
            {
                dilationVector.y = ieee754::flipSign(dilationVector.y, true);
            }
            else if (corner.x == 1.0f && corner.y == 0.0f)
            {
                uvOffset.y = -uvOffset.y;
            }

            const float2 uv = corner + uvOffset;
            outV.setImageUV(uv);

            pfloat64_t2 worldSpaceExtentsYAxisFlipped;
            worldSpaceExtentsYAxisFlipped.x = worldSpaceExtents.x;
            worldSpaceExtentsYAxisFlipped.y = ieee754::flipSign(worldSpaceExtents.y, true);
            const pfloat64_t2 vtxPos = topLeft + worldSpaceExtentsYAxisFlipped * _static_cast<pfloat64_t2>(corner);
            const pfloat64_t2 dilatedVtxPos = vtxPos + dilationVector;

            float2 ndcVtxPos = _static_cast<float2>(transformPointNdc(clipProjectionData.projectionToNDC, dilatedVtxPos));
            outV.position = float4(ndcVtxPos, 0.0f, 1.0f);
        }
        else if (objType == ObjectType::STREAMED_IMAGE)
        {
            pfloat64_t2 topLeft = vk::RawBufferLoad<pfloat64_t2>(globals.pointers.geometryBuffer + drawObj.geometryAddress, 8u);
            float32_t2 dirU = vk::RawBufferLoad<float32_t2>(globals.pointers.geometryBuffer + drawObj.geometryAddress + sizeof(pfloat64_t2), 4u);
            float32_t aspectRatio = vk::RawBufferLoad<float32_t>(globals.pointers.geometryBuffer + drawObj.geometryAddress + sizeof(pfloat64_t2) + sizeof(float2), 4u);
            uint32_t textureID = vk::RawBufferLoad<uint32_t>(globals.pointers.geometryBuffer + drawObj.geometryAddress + sizeof(pfloat64_t2) + sizeof(float2) + sizeof(float), 4u);

            const float32_t2 dirV = float32_t2(dirU.y, -dirU.x) * aspectRatio;
            const float2 ndcTopLeft = _static_cast<float2>(transformPointNdc(clipProjectionData.projectionToNDC, topLeft));
            const float2 ndcDirU = _static_cast<float2>(transformVectorNdc(clipProjectionData.projectionToNDC, _static_cast<pfloat64_t2>(dirU)));
            const float2 ndcDirV = _static_cast<float2>(transformVectorNdc(clipProjectionData.projectionToNDC, _static_cast<pfloat64_t2>(dirV)));

            float2 corner = float2(bool2(vertexIdx & 0x1u, vertexIdx >> 1));
            float2 uv = corner; // non-dilated
        
            float2 ndcCorner = ndcTopLeft + corner.x * ndcDirU + corner.y * ndcDirV;
        
            outV.position = float4(ndcCorner, 0.f, 1.f);
            outV.setImageUV(uv);
            outV.setImageTextureId(textureID);
        }

    // Make the cage fullscreen for testing: 
#if 0
        if (vertexIdx == 0u)
            outV.position = float4(-1, -1, 0, 1);
        else if (vertexIdx == 1u)
            outV.position = float4(-1, +1, 0, 1);
        else if (vertexIdx == 2u)
            outV.position = float4(+1, -1, 0, 1);
        else if (vertexIdx == 3u)
            outV.position = float4(+1, +1, 0, 1);
#endif
    }
    outV.clip = float4(outV.position.x - clipProjectionData.minClipNDC.x, outV.position.y - clipProjectionData.minClipNDC.y, clipProjectionData.maxClipNDC.x - outV.position.x, clipProjectionData.maxClipNDC.y - outV.position.y);
    return outV;
}
