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
    return nbl::hlsl::shapes::QuadraticBezier<float>::construct(p0, p1, p2).evaluate(t);
}

ClipProjectionData getClipProjectionData(in MainObject mainObj)
{
    if (mainObj.clipProjectionIdx != InvalidClipProjectionIdx)
    {
        return customClipProjections[mainObj.clipProjectionIdx];
    }
    else
    {
        return globals.defaultClipProjection;
    }
}

double2 transformPointNdc(float64_t3x3 transformation, double2 point2d)
{
    return mul(transformation, float64_t3(point2d, 1)).xy;
}
double2 transformVectorNdc(float64_t3x3 transformation, double2 vector2d)
{
    return mul(transformation, float64_t3(vector2d, 0)).xy;
}
float2 transformPointScreenSpace(float64_t3x3 transformation, double2 point2d) 
{
    double2 ndc = transformPointNdc(transformation, point2d);
    return (float2)((ndc + 1.0) * 0.5 * globals.resolution);
}
float4 transformFromSreenSpaceToNdc(float2 pos)
{
    return float4((pos.xy / globals.resolution) * 2.0 - 1.0, 0.0f, 1.0f);
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
    outV.position.z = 0.0;
    outV.data1 = uint4(0, 0, 0, 0);
    outV.data2 = float4(0, 0, 0, 0);
    outV.data3 = float4(0, 0, 0, 0);
    outV.data4 = float4(0, 0, 0, 0);
    outV.interp_data5 = float2(0, 0);
    outV.setObjType(objType);
    outV.setMainObjectIdx(drawObj.mainObjIndex);

    // We only need these for Outline type objects like lines and bezier curves
    MainObject mainObj = mainObjects[drawObj.mainObjIndex];
    LineStyle lineStyle = lineStyles[mainObj.styleIdx];
    ClipProjectionData clipProjectionData = getClipProjectionData(mainObj);
    const float screenSpaceLineWidth = lineStyle.screenSpaceLineWidth + float(lineStyle.worldSpaceLineWidth * globals.screenToWorldRatio);
    const float antiAliasedLineWidth = screenSpaceLineWidth + globals.antiAliasingFactor * 2.0f;

    if (objType == ObjectType::LINE)
    {
        outV.setLineThickness(screenSpaceLineWidth / 2.0f);

        const double3x3 transformation = clipProjectionData.projectionToNDC;

        double2 points[2u];
        points[0u] = vk::RawBufferLoad<double2>(drawObj.geometryAddress, 8u);
        points[1u] = vk::RawBufferLoad<double2>(drawObj.geometryAddress + sizeof(LinePointInfo), 8u);

        const float phaseShift = vk::RawBufferLoad<float>(drawObj.geometryAddress + sizeof(double2), 8u);
        outV.setCurrentPhaseShift(phaseShift);

        float2 transformedPoints[2u];
        for (uint i = 0u; i < 2u; ++i)
        {
            transformedPoints[i] = transformPointScreenSpace(clipProjectionData.projectionToNDC, points[i]);
        }

        const float2 lineVector = normalize(transformedPoints[1u] - transformedPoints[0u]);
        const float2 normalToLine = float2(-lineVector.y, lineVector.x);

        if (vertexIdx == 0u || vertexIdx == 1u)
        {
            // work in screen space coordinates because of fixed pixel size
            outV.position.xy = transformedPoints[0u]
                + normalToLine * (((float)vertexIdx - 0.5f) * antiAliasedLineWidth)
                - lineVector * antiAliasedLineWidth * 0.5f;
        }
        else // if (vertexIdx == 2u || vertexIdx == 3u)
        {
            // work in screen space coordinates because of fixed pixel size
            outV.position.xy = transformedPoints[1u]
                + normalToLine * (((float)vertexIdx - 2.5f) * antiAliasedLineWidth)
                + lineVector * antiAliasedLineWidth * 0.5f;
        }

        outV.setLineStart(transformedPoints[0u]);
        outV.setLineEnd(transformedPoints[1u]);

        outV.position = transformFromSreenSpaceToNdc(outV.position.xy);
    }
    else if (objType == ObjectType::QUAD_BEZIER)
    {
        outV.setLineThickness(screenSpaceLineWidth / 2.0f);

        double2 points[3u];
        points[0u] = vk::RawBufferLoad<double2>(drawObj.geometryAddress, 8u);
        points[1u] = vk::RawBufferLoad<double2>(drawObj.geometryAddress + sizeof(double2), 8u);
        points[2u] = vk::RawBufferLoad<double2>(drawObj.geometryAddress + sizeof(double2) * 2u, 8u);

        const float phaseShift = vk::RawBufferLoad<float>(drawObj.geometryAddress + sizeof(double2) * 3u, 8u);
        outV.setCurrentPhaseShift(phaseShift);

        // transform these points into screen space and pass to fragment
        float2 transformedPoints[3u];
        for (uint i = 0u; i < 3u; ++i)
        {
            transformedPoints[i] = transformPointScreenSpace(clipProjectionData.projectionToNDC, points[i]);
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
        //  the reason for division by 4.0f is 1. screenSpaceLineWidth is expanded on both sides and the fact that diameter/2=radius, 
        const bool noCurvature = abs(dot(normalize(vectorAB), normalize(vectorAC)) - 1.0f) < exp2(-10.0f);
        if (MaxCurvature * screenSpaceLineWidth > 4.0f || noCurvature)
        {
            //OBB Fallback
            float2 obbV0;
            float2 obbV1;
            float2 obbV2;
            float2 obbV3;
            quadraticBezier.OBBAligned(screenSpaceLineWidth / 2.0f, obbV0, obbV1, obbV2, obbV3);
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
            
            float2 middleExteriorPoint = midPos - midNormal * screenSpaceLineWidth / 2.0f;
            
            
            float2 leftTangent = normalize(BezierTangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], optimalT));
            float2 leftNormal = normalize(float2(-leftTangent.y, leftTangent.x)) * flip;
            float2 leftExteriorPoint = QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], optimalT) - leftNormal * screenSpaceLineWidth / 2.0f;
            float2 exterior0 = nbl::hlsl::shapes::util::LineLineIntersection(middleExteriorPoint, midTangent, leftExteriorPoint, leftTangent);
            
            float2 rightTangent = normalize(BezierTangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 1.0f-optimalT));
            float2 rightNormal = normalize(float2(-rightTangent.y, rightTangent.x)) * flip;
            float2 rightExteriorPoint = QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 1.0f-optimalT) - rightNormal * screenSpaceLineWidth / 2.0f;
            float2 exterior1 = nbl::hlsl::shapes::util::LineLineIntersection(middleExteriorPoint, midTangent, rightExteriorPoint, rightTangent);

            // Interiors
            {
                float2 tangent = normalize(BezierTangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.286f));
                float2 normal = normalize(float2(-tangent.y, tangent.x)) * flip;
                interior0 = QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.286) + normal * screenSpaceLineWidth / 2.0f;
            }
            {
                float2 tangent = normalize(BezierTangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.714f));
                float2 normal = normalize(float2(-tangent.y, tangent.x)) * flip;
                interior1 = QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.714f) + normal * screenSpaceLineWidth / 2.0f;
            }

            if (subsectionIdx == 0u)
            {
                float2 endPointTangent = normalize(transformedPoints[1u]-transformedPoints[0u]);
                float2 endPointNormal = float2(-endPointTangent.y, endPointTangent.x) * flip;
                float2 endPointExterior = transformedPoints[0u] - endPointTangent * screenSpaceLineWidth / 2.0f;

                if (vertexIdx == 0u)
                    outV.position = float4(nbl::hlsl::shapes::util::LineLineIntersection(leftExteriorPoint, leftTangent, endPointExterior, endPointNormal), 0.0, 1.0f);
                else if (vertexIdx == 1u)
                    outV.position = float4(transformedPoints[0u] + endPointNormal * screenSpaceLineWidth / 2.0f - endPointTangent * screenSpaceLineWidth / 2.0f, 0.0, 1.0f);
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
                float2 endPointTangent = normalize(transformedPoints[2u]-transformedPoints[1u]);
                float2 endPointNormal = float2(-endPointTangent.y, endPointTangent.x) * flip;
                float2 endPointExterior = transformedPoints[2u] + endPointTangent * screenSpaceLineWidth / 2.0f;

                if (vertexIdx == 0u)
                    outV.position = float4(nbl::hlsl::shapes::util::LineLineIntersection(rightExteriorPoint, rightTangent, endPointExterior, endPointNormal), 0.0, 1.0f);
                else if (vertexIdx == 1u)
                    outV.position = float4(transformedPoints[2u] + endPointNormal * screenSpaceLineWidth / 2.0f + endPointTangent * screenSpaceLineWidth / 2.0f, 0.0, 1.0f);
                else if (vertexIdx == 2u)
                    outV.position = float4(exterior1, 0.0, 1.0f);
                else if (vertexIdx == 3u)
                    outV.position = float4(interior1, 0.0, 1.0f);
            }
        }

        outV.position.xy = (outV.position.xy / globals.resolution) * 2.0 - 1.0;
    }
    else if (objType == ObjectType::CURVE_BOX)
    {
        outV.setLineThickness(screenSpaceLineWidth / 2.0f);

        CurveBox curveBox;
        curveBox.aabbMin = vk::RawBufferLoad<double2>(drawObj.geometryAddress, 8u);
        curveBox.aabbMax = vk::RawBufferLoad<double2>(drawObj.geometryAddress + sizeof(double2), 8u);
        for (uint32_t i = 0; i < 3; i ++)
        {
            curveBox.curveMin[i] = vk::RawBufferLoad<float32_t2>(drawObj.geometryAddress + sizeof(double2) * 2 + sizeof(float32_t2) * i, 4u);
            curveBox.curveMax[i] = vk::RawBufferLoad<float32_t2>(drawObj.geometryAddress + sizeof(double2) * 2 + sizeof(float32_t2) * (3 + i), 4u);
        }

        // TODO: better name?
        // TODO: can we use floats instead of doubles for every dilation and ndc things except the main box values
        const double2 ndcBoxAxisX = transformVectorNdc(clipProjectionData.projectionToNDC, double2(curveBox.aabbMax.x, curveBox.aabbMin.y) - curveBox.aabbMin);
        const double2 ndcBoxAxisY = transformVectorNdc(clipProjectionData.projectionToNDC, double2(curveBox.aabbMin.x, curveBox.aabbMax.y) - curveBox.aabbMin);

        const double2 screenSpaceAabbExtents = double2(length(ndcBoxAxisX * double2(globals.resolution)) / 2.0, length(ndcBoxAxisY * double2(globals.resolution)) / 2.0);

        // we could use something like  this to compute screen space change over minor/major change and avoid ddx(minor), ddy(major) in frag shader (the code below doesn't account for rotation)
        outV.setCurveBoxScreenSpaceSize(float2(screenSpaceAabbExtents));
        
        const float pixelsToIncreaseOnEachSide = globals.antiAliasingFactor + 1.0;
        const double2 dilateRate = pixelsToIncreaseOnEachSide / screenSpaceAabbExtents;
        const double2 dilatationFactor = 1.0 + 2.0 * dilateRate;
        // undilatedMaxCornerNDC stores the quad's UVs:
        // (-1,-1)|--|(1,-1)
        //        |  |
        // (-1,1) |--|(1,1)
        const float2 undilatedCorner = float2(bool2(vertexIdx & 0x1u, vertexIdx >> 1));
        const float2 undilatedCornerNDC = float2(undilatedCorner * 2.0 - 1.0);
        // Dilate the UVs
        const float2 maxCorner = float2((undilatedCornerNDC * dilatationFactor + 1.0) * 0.5);
        
        // vx/vy are vectors in direction of the box's axes and their length is equal to X pixels (X = globals.antiAliasingFactor + 1.0)
        // and we use them for dilation of X pixels in ndc space by adding them to the currentCorner in NDC space 
        const double2 vx = ndcBoxAxisX * dilateRate.x;
        const double2 vy = ndcBoxAxisY * dilateRate.y;
        const double2 offsetVec = vx * undilatedCornerNDC.x + vy * undilatedCornerNDC.y; // (0, 0) should do -vx-vy and (1, 1) should do +vx+vy

        // doing interpolation this way to ensure correct endpoints and 0 and 1, we can alternatively use branches to set current corner based on vertexIdx
        const double2 currentCorner = curveBox.aabbMin * (1.0 - undilatedCorner) + curveBox.aabbMax * undilatedCorner;
        const float2 coord = (float2) (transformPointNdc(clipProjectionData.projectionToNDC, currentCorner) + offsetVec);

        outV.position = float4(coord, 0.f, 1.f);
 
        const uint major = (uint)SelectedMajorAxis;
        const uint minor = 1-major;

        // A, B & C get converted from unorm to [0, 1]
        // A & B get converted from [0,1] to [-2, 2]
        nbl::hlsl::shapes::Quadratic<float> curveMin = nbl::hlsl::shapes::Quadratic<float>::construct(
            curveBox.curveMin[0], curveBox.curveMin[1], curveBox.curveMin[2]);
        nbl::hlsl::shapes::Quadratic<float> curveMax = nbl::hlsl::shapes::Quadratic<float>::construct(
            curveBox.curveMax[0], curveBox.curveMax[1], curveBox.curveMax[2]);

        outV.setMinorBBoxUv(maxCorner[minor]);
        outV.setMajorBBoxUv(maxCorner[major]);

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
    else if (objType == ObjectType::POLYLINE_CONNECTOR)
    {
        const float FLOAT_INF = nbl::hlsl::numeric_limits<float>::infinity;
        const float4 INVALID_VERTEX = float4(FLOAT_INF, FLOAT_INF, FLOAT_INF, FLOAT_INF);

        if (lineStyle.isRoadStyleFlag)
        {
            const float sdfLineThickness = screenSpaceLineWidth / 2.0f;
            outV.setLineThickness(sdfLineThickness);
                
            const double2 circleCenter = vk::RawBufferLoad<double2>(drawObj.geometryAddress, 8u);
            const float2 v = vk::RawBufferLoad<float2>(drawObj.geometryAddress + sizeof(double2), 8u);
            const float cosHalfAngleBetweenNormals = vk::RawBufferLoad<float>(drawObj.geometryAddress + sizeof(double2) + sizeof(float2), 8u);

            const float2 circleCenterScreenSpace = transformPointScreenSpace(clipProjectionData.projectionToNDC, circleCenter);
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
                const float2 V1 = normalize(mul(v, rotationMatrix)) * antiAliasedLineWidth;
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
                float2 intersectionPoint = v * antiAliasedLineWidth;
                intersectionPoint += circleCenterScreenSpace;
                outV.position = float4(intersectionPoint, 0.0f, 1.0f);
            }
            else if (vertexIdx == 3u)
            {
                const float2 V2 = normalize(mul(rotationMatrix, v)) * antiAliasedLineWidth;
                const float2 screenSpaceV2 = circleCenterScreenSpace + V2;
                outV.position = float4(screenSpaceV2, 0.0f, 1.0f);
            }

            outV.position = transformFromSreenSpaceToNdc(outV.position.xy);
        }
        else
        {
            outV.position = INVALID_VERTEX;
        }
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
