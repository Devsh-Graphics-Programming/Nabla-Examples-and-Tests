#pragma shader_stage(vertex)

#include "common.hlsl"
#include <nbl/builtin/hlsl/shapes/beziers.hlsl>
#include <nbl/builtin/hlsl/math/equations/quadratic.hlsl>
#include <nbl/builtin/hlsl/limits.hlsl>
#include <nbl/builtin/hlsl/algorithm.hlsl>
#include <nbl/builtin/hlsl/jit/device_capabilities.hlsl>

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
    return nbl::hlsl::shapes::QuadraticBezier<float>::construct(p0, p1, p2).evaluate(t);
}

ClipProjectionData getClipProjectionData(in MainObject mainObj)
{
    if (mainObj.clipProjectionAddress != InvalidClipProjectionAddress)
    {
        ClipProjectionData ret;
        ret.projectionToNDC = vk::RawBufferLoad<float64_t3x3>(mainObj.clipProjectionAddress, 8u);
        ret.minClipNDC      = vk::RawBufferLoad<float32_t2>(mainObj.clipProjectionAddress + sizeof(float64_t3x3), 8u);
        ret.maxClipNDC      = vk::RawBufferLoad<float32_t2>(mainObj.clipProjectionAddress + sizeof(float64_t3x3) + sizeof(float32_t2), 8u);
        return ret;
    }
    else
    {
        return globals.defaultClipProjection;
    }
}

float32_t2 transformPointScreenSpace(float64_t3x3 transformation, uint32_t2 resolution, float64_t2 point2d) 
{
    float64_t2 ndc = transformPointNdc(transformation, point2d);
    return (float32_t2)((ndc + 1.0) * 0.5 * resolution);
}
float32_t2 transformFromSreenSpaceToNdc(float32_t2 pos, uint32_t2 resolution)
{
    return float32_t2((pos / (float32_t2)resolution) * 2.0f - 1.0f);
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

PSInput main(uint vertexID : SV_VertexID)
{
    const uint vertexIdx = vertexID & 0x3u;
    const uint objectID = vertexID >> 2;

    DrawObject drawObj = drawObjects[objectID];

    ObjectType objType = (ObjectType)(drawObj.type_subsectionIdx & 0x0000FFFF);
    uint32_t subsectionIdx = drawObj.type_subsectionIdx >> 16;
    PSInput outV;

    // Default Initialize PS Input
    outV.position.zw = float2(0.0, 1.0);
    outV.data1 = uint4(0, 0, 0, 0);
    outV.data2 = float4(0, 0, 0, 0);
    outV.data3 = float4(0, 0, 0, 0);
    outV.data4 = float4(0, 0, 0, 0);
    outV.interp_data5 = float2(0, 0);
    outV.setObjType(objType);
    outV.setMainObjectIdx(drawObj.mainObjIndex);
    
    MainObject mainObj = mainObjects[drawObj.mainObjIndex];
    ClipProjectionData clipProjectionData = getClipProjectionData(mainObj);
    
    // We only need these for Outline type objects like lines and bezier curves
    if (objType == ObjectType::LINE || objType == ObjectType::QUAD_BEZIER || objType == ObjectType::POLYLINE_CONNECTOR)
    {
        LineStyle lineStyle = lineStyles[mainObj.styleIdx];

        // Width is on both sides, thickness is one one side of the curve (div by 2.0f)
        const float screenSpaceLineWidth = lineStyle.screenSpaceLineWidth + float(lineStyle.worldSpaceLineWidth * globals.screenToWorldRatio);
        const float antiAliasedLineThickness = screenSpaceLineWidth * 0.5f + globals.antiAliasingFactor;
        const float sdfLineThickness = screenSpaceLineWidth / 2.0f;
        outV.setLineThickness(sdfLineThickness);
        outV.setCurrentWorldToScreenRatio((float)(2.0 / (clipProjectionData.projectionToNDC[0][0] * globals.resolution.x)));

        if (objType == ObjectType::LINE)
        {
            double2 points[2u];
            points[0u] = vk::RawBufferLoad<double2>(drawObj.geometryAddress, 8u);
            points[1u] = vk::RawBufferLoad<double2>(drawObj.geometryAddress + sizeof(LinePointInfo), 8u);

            const float phaseShift = vk::RawBufferLoad<float>(drawObj.geometryAddress + sizeof(double2), 8u);
            const float patternStretch = vk::RawBufferLoad<float>(drawObj.geometryAddress + sizeof(double2) + sizeof(float), 8u);
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

            outV.position.xy = transformFromSreenSpaceToNdc(outV.position.xy, globals.resolution);
        }
        else if (objType == ObjectType::QUAD_BEZIER)
        {
            double2 points[3u];
            points[0u] = vk::RawBufferLoad<double2>(drawObj.geometryAddress, 8u);
            points[1u] = vk::RawBufferLoad<double2>(drawObj.geometryAddress + sizeof(double2), 8u);
            points[2u] = vk::RawBufferLoad<double2>(drawObj.geometryAddress + sizeof(double2) * 2u, 8u);

            const float phaseShift = vk::RawBufferLoad<float>(drawObj.geometryAddress + sizeof(double2) * 3u, 8u);
            const float patternStretch = vk::RawBufferLoad<float>(drawObj.geometryAddress + sizeof(double2) * 3u + sizeof(float), 8u);
            outV.setCurrentPhaseShift(phaseShift);
            outV.setPatternStretch(patternStretch);

            // transform these points into screen space and pass to fragment
            float2 transformedPoints[3u];
            for (uint i = 0u; i < 3u; ++i)
            {
                transformedPoints[i] = transformPointScreenSpace(clipProjectionData.projectionToNDC, globals.resolution, points[i]);
            }

            nbl::hlsl::shapes::QuadraticBezier<float> quadraticBezier = nbl::hlsl::shapes::QuadraticBezier<float>::construct(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u]);
            nbl::hlsl::shapes::Quadratic<float> quadratic = nbl::hlsl::shapes::Quadratic<float>::constructFromBezier(quadraticBezier);
            nbl::hlsl::shapes::Quadratic<float>::ArcLengthCalculator preCompData = nbl::hlsl::shapes::Quadratic<float>::ArcLengthCalculator::construct(quadratic);

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

                //Whether or not to flip the the interior cage nodes
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

                //Internal cage points
                float2 interior0;
                float2 interior1;

                float2 middleExteriorPoint = midPos - midNormal * antiAliasedLineThickness;


                float2 leftTangent = normalize(BezierTangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], optimalT));
                float2 leftNormal = normalize(float2(-leftTangent.y, leftTangent.x)) * flip;
                float2 leftExteriorPoint = QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], optimalT) - leftNormal * antiAliasedLineThickness;
                float2 exterior0 = nbl::hlsl::shapes::util::LineLineIntersection<float>(middleExteriorPoint, midTangent, leftExteriorPoint, leftTangent);

                float2 rightTangent = normalize(BezierTangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 1.0f - optimalT));
                float2 rightNormal = normalize(float2(-rightTangent.y, rightTangent.x)) * flip;
                float2 rightExteriorPoint = QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 1.0f - optimalT) - rightNormal * antiAliasedLineThickness;
                float2 exterior1 = nbl::hlsl::shapes::util::LineLineIntersection<float>(middleExteriorPoint, midTangent, rightExteriorPoint, rightTangent);

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
                        outV.position = float4(nbl::hlsl::shapes::util::LineLineIntersection<float>(leftExteriorPoint, leftTangent, endPointExterior, endPointNormal), 0.0, 1.0f);
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
                        outV.position = float4(nbl::hlsl::shapes::util::LineLineIntersection<float>(rightExteriorPoint, rightTangent, endPointExterior, endPointNormal), 0.0, 1.0f);
                    else if (vertexIdx == 1u)
                        outV.position = float4(transformedPoints[2u] + endPointNormal * antiAliasedLineThickness + endPointTangent * antiAliasedLineThickness, 0.0, 1.0f);
                    else if (vertexIdx == 2u)
                        outV.position = float4(exterior1, 0.0, 1.0f);
                    else if (vertexIdx == 3u)
                        outV.position = float4(interior1, 0.0, 1.0f);
                }
            }

            outV.position.xy = (outV.position.xy / globals.resolution) * 2.0 - 1.0;
        }
        else if (objType == ObjectType::POLYLINE_CONNECTOR)
        {
            const float FLOAT_INF = nbl::hlsl::numeric_limits<float>::infinity;
            const float4 INVALID_VERTEX = float4(FLOAT_INF, FLOAT_INF, FLOAT_INF, FLOAT_INF);

            if (lineStyle.isRoadStyleFlag)
            {
                const double2 circleCenter = vk::RawBufferLoad<double2>(drawObj.geometryAddress, 8u);
                const float2 v = vk::RawBufferLoad<float2>(drawObj.geometryAddress + sizeof(double2), 8u);
                const float cosHalfAngleBetweenNormals = vk::RawBufferLoad<float>(drawObj.geometryAddress + sizeof(double2) + sizeof(float2), 8u);

                const float2 circleCenterScreenSpace = transformPointScreenSpace(clipProjectionData.projectionToNDC, globals.resolution, circleCenter);
                outV.setPolylineConnectorCircleCenter(circleCenterScreenSpace);

                // Find other miter vertices
                const float sinHalfAngleBetweenNormals = sqrt(1.0f - (cosHalfAngleBetweenNormals * cosHalfAngleBetweenNormals));
                const float32_t2x2 rotationMatrix = float32_t2x2(cosHalfAngleBetweenNormals, -sinHalfAngleBetweenNormals, sinHalfAngleBetweenNormals, cosHalfAngleBetweenNormals);

                // Pass the precomputed trapezoid values for the sdf
                {
                    float vLen = length(v);
                    float2 intersectionDirection = v / vLen;

                    float longBase = sinHalfAngleBetweenNormals;
                    float shortBase = max((vLen - globals.miterLimit) * cosHalfAngleBetweenNormals / sinHalfAngleBetweenNormals, 0.0);
                    // height of the trapezoid / triangle
                    float hLen = min(globals.miterLimit, vLen);

                    outV.setPolylineConnectorTrapezoidStart(-1.0 * intersectionDirection * sdfLineThickness);
                    outV.setPolylineConnectorTrapezoidEnd(intersectionDirection * hLen * sdfLineThickness);
                    outV.setPolylineConnectorTrapezoidLongBase(sinHalfAngleBetweenNormals * ((1.0 + vLen) / (vLen - cosHalfAngleBetweenNormals)) * sdfLineThickness);
                    outV.setPolylineConnectorTrapezoidShortBase(shortBase * sdfLineThickness);
                }

                if (vertexIdx == 0u)
                {
                    const float2 V1 = normalize(mul(v, rotationMatrix)) * antiAliasedLineThickness * 2.0f;
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
                    float2 intersectionPoint = v * antiAliasedLineThickness * 2.0f;
                    intersectionPoint += circleCenterScreenSpace;
                    outV.position = float4(intersectionPoint, 0.0f, 1.0f);
                }
                else if (vertexIdx == 3u)
                {
                    const float2 V2 = normalize(mul(rotationMatrix, v)) * antiAliasedLineThickness * 2.0f;
                    const float2 screenSpaceV2 = circleCenterScreenSpace + V2;
                    outV.position = float4(screenSpaceV2, 0.0f, 1.0f);
                }

                outV.position.xy = transformFromSreenSpaceToNdc(outV.position.xy, globals.resolution);
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
        curveBox.aabbMin = vk::RawBufferLoad<double2>(drawObj.geometryAddress, 8u);
        curveBox.aabbMax = vk::RawBufferLoad<double2>(drawObj.geometryAddress + sizeof(double2), 8u);
        for (uint32_t i = 0; i < 3; i ++)
        {
            curveBox.curveMin[i] = vk::RawBufferLoad<float32_t2>(drawObj.geometryAddress + sizeof(double2) * 2 + sizeof(float32_t2) * i, 4u);
            curveBox.curveMax[i] = vk::RawBufferLoad<float32_t2>(drawObj.geometryAddress + sizeof(double2) * 2 + sizeof(float32_t2) * (3 + i), 4u);
        }

        const float2 ndcAxisU = (float2)transformVectorNdc(clipProjectionData.projectionToNDC, double2(curveBox.aabbMax.x, curveBox.aabbMin.y) - curveBox.aabbMin);
        const float2 ndcAxisV = (float2)transformVectorNdc(clipProjectionData.projectionToNDC, double2(curveBox.aabbMin.x, curveBox.aabbMax.y) - curveBox.aabbMin);

        const float2 screenSpaceAabbExtents = float2(length(ndcAxisU * float2(globals.resolution)) / 2.0, length(ndcAxisV * float2(globals.resolution)) / 2.0);

        // we could use something like  this to compute screen space change over minor/major change and avoid ddx(minor), ddy(major) in frag shader (the code below doesn't account for rotation)
        outV.setCurveBoxScreenSpaceSize(float2(screenSpaceAabbExtents));
        
        const float2 undilatedCorner = float2(bool2(vertexIdx & 0x1u, vertexIdx >> 1));
        
        // We don't dilate on AMD (= no fragShaderInterlock)
        const float pixelsToIncreaseOnEachSide = globals.antiAliasingFactor + 1.0;
        const float2 dilateRate = pixelsToIncreaseOnEachSide / screenSpaceAabbExtents; // float sufficient to hold the dilate rect? 
        float2 dilateVec;
        float2 dilatedUV;
        dilateHatch<nbl::hlsl::jit::device_capabilities::fragmentShaderPixelInterlock>(dilateVec, dilatedUV, undilatedCorner, dilateRate, ndcAxisU, ndcAxisV);

        // doing interpolation this way to ensure correct endpoints and 0 and 1, we can alternatively use branches to set current corner based on vertexIdx
        const double2 currentCorner = curveBox.aabbMin * (1.0 - undilatedCorner) + curveBox.aabbMax * undilatedCorner;
        const float2 coord = (float2) (transformPointNdc(clipProjectionData.projectionToNDC, currentCorner) + dilateVec);

        outV.position = float4(coord, 0.f, 1.f);
 
        const uint major = (uint)SelectedMajorAxis;
        const uint minor = 1-major;

        // A, B & C get converted from unorm to [0, 1]
        // A & B get converted from [0,1] to [-2, 2]
        nbl::hlsl::shapes::Quadratic<float> curveMin = nbl::hlsl::shapes::Quadratic<float>::construct(
            curveBox.curveMin[0], curveBox.curveMin[1], curveBox.curveMin[2]);
        nbl::hlsl::shapes::Quadratic<float> curveMax = nbl::hlsl::shapes::Quadratic<float>::construct(
            curveBox.curveMax[0], curveBox.curveMax[1], curveBox.curveMax[2]);

        outV.setMinorBBoxUV(dilatedUV[minor]);
        outV.setMajorBBoxUV(dilatedUV[major]);

        outV.setCurveMinMinor(nbl::hlsl::math::equations::Quadratic<float>::construct(
            curveMin.A[minor], 
            curveMin.B[minor], 
            curveMin.C[minor]));
        outV.setCurveMinMajor(nbl::hlsl::math::equations::Quadratic<float>::construct(
            curveMin.A[major], 
            curveMin.B[major], 
            curveMin.C[major]));

        outV.setCurveMaxMinor(nbl::hlsl::math::equations::Quadratic<float>::construct(
            curveMax.A[minor], 
            curveMax.B[minor], 
            curveMax.C[minor]));
        outV.setCurveMaxMajor(nbl::hlsl::math::equations::Quadratic<float>::construct(
            curveMax.A[major], 
            curveMax.B[major], 
            curveMax.C[major]));

        //nbl::hlsl::math::equations::Quadratic<float> curveMinRootFinding = nbl::hlsl::math::equations::Quadratic<float>::construct(
        //    curveMin.A[major], 
        //    curveMin.B[major], 
        //    curveMin.C[major] - maxCorner[major]);
        //nbl::hlsl::math::equations::Quadratic<float> curveMaxRootFinding = nbl::hlsl::math::equations::Quadratic<float>::construct(
        //    curveMax.A[major], 
        //    curveMax.B[major], 
        //    curveMax.C[major] - maxCorner[major]);
        //outV.setMinCurvePrecomputedRootFinders(PrecomputedRootFinder<float>::construct(curveMinRootFinding));
        //outV.setMaxCurvePrecomputedRootFinders(PrecomputedRootFinder<float>::construct(curveMaxRootFinding));
    }
    else if (objType == ObjectType::FONT_GLYPH)
    {
        LineStyle lineStyle = lineStyles[mainObj.styleIdx];
        const float italicTiltSlope = lineStyle.screenSpaceLineWidth; // aliased text style member with line style
        
        GlyphInfo glyphInfo;
        glyphInfo.topLeft = vk::RawBufferLoad<double2>(drawObj.geometryAddress, 8u);
        glyphInfo.dirU = vk::RawBufferLoad<float32_t2>(drawObj.geometryAddress + sizeof(double2), 4u);
        glyphInfo.aspectRatio = vk::RawBufferLoad<float32_t>(drawObj.geometryAddress + sizeof(double2) + sizeof(float2), 4u);
        glyphInfo.minUV_textureID_packed = vk::RawBufferLoad<uint32_t>(drawObj.geometryAddress + sizeof(double2) + sizeof(float2) + sizeof(float), 4u);

        float32_t2 minUV = glyphInfo.getMinUV();
        uint16_t textureID = glyphInfo.getTextureID();

        const float32_t2 dirV = float32_t2(glyphInfo.dirU.y, -glyphInfo.dirU.x) * glyphInfo.aspectRatio;
        const float2 screenTopLeft = (float2) transformPointNdc(clipProjectionData.projectionToNDC, glyphInfo.topLeft);
        const float2 screenDirU = (float2) transformVectorNdc(clipProjectionData.projectionToNDC, glyphInfo.dirU);
        const float2 screenDirV = (float2) transformVectorNdc(clipProjectionData.projectionToNDC, dirV);

        const float2 corner = float2(bool2(vertexIdx & 0x1u, vertexIdx >> 1)); // corners of square from (0, 0) to (1, 1)
        const float2 undilatedCornerNDC = corner * 2.0 - 1.0; // corners of square from (-1, -1) to (1, 1)
        
        const float2 screenSpaceAabbExtents = float2(length(screenDirU * float2(globals.resolution)) / 2.0, length(screenDirV * float2(globals.resolution)) / 2.0);
        const float pixelsToIncreaseOnEachSide = globals.antiAliasingFactor + 1.0;
        const float2 dilateRate = (float2)(pixelsToIncreaseOnEachSide / screenSpaceAabbExtents);

        const float2 vx = screenDirU * dilateRate.x;
        const float2 vy = screenDirV * dilateRate.y;
        const float2 offsetVec = vx * undilatedCornerNDC.x + vy * undilatedCornerNDC.y;
        float2 coord = screenTopLeft + corner.x * screenDirU + corner.y * screenDirV + offsetVec;

        if (corner.y == 0 && italicTiltSlope > 0.0f)
            coord += normalize(screenDirU) * length(screenDirV) * italicTiltSlope;
        
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
    else if (objType == ObjectType::IMAGE)
    {
        float64_t2 topLeft = vk::RawBufferLoad<double2>(drawObj.geometryAddress, 8u);
        float32_t2 dirU = vk::RawBufferLoad<float32_t2>(drawObj.geometryAddress + sizeof(double2), 4u);
        float32_t aspectRatio = vk::RawBufferLoad<float32_t>(drawObj.geometryAddress + sizeof(double2) + sizeof(float2), 4u);
        uint32_t textureID = vk::RawBufferLoad<uint32_t>(drawObj.geometryAddress + sizeof(double2) + sizeof(float2) + sizeof(float), 4u);

        const float32_t2 dirV = float32_t2(dirU.y, -dirU.x) * aspectRatio;
        const float2 ndcTopLeft = (float2) transformPointNdc(clipProjectionData.projectionToNDC, topLeft);
        const float2 ndcDirU = (float2) transformVectorNdc(clipProjectionData.projectionToNDC, dirU);
        const float2 ndcDirV = (float2) transformVectorNdc(clipProjectionData.projectionToNDC, dirV);

        float2 corner = float2(bool2(vertexIdx & 0x1u, vertexIdx >> 1));
        float2 uv = corner; // non-dilated
        
        float2 ndcCorner = ndcTopLeft + corner.x * ndcDirU + corner.y * ndcDirV;
        
        outV.position = float4(ndcCorner, 0.f, 1.f);
        outV.setImageUV(uv);
        outV.setImageTextureId(textureID);
    }


// Make the cage fullscreen for testing: 
#if 0
    // disabled for object of POLYLINE_CONNECTOR type, since miters would cover whole screen
    if(objType != ObjectType::POLYLINE_CONNECTOR)
    {
        if (vertexIdx == 0u)
            outV.position = float4(-1, -1, 0, 1);
        else if (vertexIdx == 1u)
            outV.position = float4(-1, +1, 0, 1);
        else if (vertexIdx == 2u)
            outV.position = float4(+1, -1, 0, 1);
        else if (vertexIdx == 3u)
            outV.position = float4(+1, +1, 0, 1);
    }
#endif

    outV.clip = float4(outV.position.x - clipProjectionData.minClipNDC.x, outV.position.y - clipProjectionData.minClipNDC.y, clipProjectionData.maxClipNDC.x - outV.position.x, clipProjectionData.maxClipNDC.y - outV.position.y);
    return outV;
}
