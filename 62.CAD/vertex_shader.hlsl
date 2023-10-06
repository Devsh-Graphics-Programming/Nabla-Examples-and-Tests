#pragma shader_stage(vertex)

#include "common.hlsl"
#include <nbl/builtin/hlsl/shapes/beziers.hlsl>
#include <nbl/builtin/hlsl/equations/quadratic.hlsl>

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

//Compute bezier in one dimension, as the OBB X and Y are at different T's
float QuadraticBezier1D(float v0, float v1, float v2, float t)
{
    float s = 1.0 - t;

    return v0 * (s * s) +
        v1 * (s * t * 2.0) +
        v2 * (t * t);
}

// Caller should make sure the lines are not parallel, i.e. cross2D(direction1, direction2) != 0, otherwise a division-by-zero will cause NaN values
float2 LineLineIntersection(float2 p1, float2 p2, float2 v1, float2 v2)
{
    // Here we're doing part of a matrix calculation because we're interested in only the intersection point not both t values
    /*
        float det = v1.y * v2.x - v1.x * v2.y;
        float2x2 inv = float2x2(v2.y, -v2.x, v1.y, -v1.x) / det;
        float2 t = mul(inv, p1 - p2);
        return p2 + mul(v2, t.y);
    */
    float denominator = v1.y * v2.x - v1.x * v2.y;
    float numerator = dot(float2(v2.y, -v2.x), p1 - p2); 

    float t = numerator / denominator;
    float2 intersectionPoint = p1 + t * v1;

    return intersectionPoint;
}

bool estimateTransformation(float2 p01, float2 p11, float2 p21, out float2 translation, out float2x2 rotation)
{
    float2 p1 = p11 - p01;
    float2 p2 = p21 - p01;

    float2 a = p2 - 2.0 * p1;
    float2 b = 2.0 * p1;

    float2 mean = a / 3.0 + b / 2.0;

    float axy = a.x * a.y;
    float bxy = a.x * b.y + b.x * a.y;
    float cxy = b.x * b.y;

    float2 aB = a * a;
    float2 bB = a * b * 2.0;
    float2 cB = b * b;

    float xy = axy / 5.0 + bxy / 4.0 + cxy / 3.0;
    float xx = aB.x / 5.0 + bB.x / 4.0 + cB.x / 3.0;
    float yy = aB.y / 5.0 + bB.y / 4.0 + cB.y / 3.0;

    float cov_00 = xx - mean.x * mean.x;
    float cov_01 = xy - mean.x * mean.y;
    float cov_11 = yy - mean.y * mean.y;

    float eigen_a = 1.0;
    float eigen_b_neghalf = -(cov_00 + cov_11) * -0.5;
    float eigen_c = (cov_00 * cov_11 - cov_01 * cov_01);

    float discr = eigen_b_neghalf * eigen_b_neghalf - eigen_a * eigen_c;
    if (discr <= 0.0)
        return false;

    discr = sqrt(discr);

    float lambda0 = (eigen_b_neghalf - discr) / eigen_a;
    float lambda1 = (eigen_b_neghalf + discr) / eigen_a;

    float2 eigenvector0 = float2(cov_01, lambda0 - cov_00);
    float2 eigenvector1 = float2(cov_01, lambda1 - cov_00);

    rotation[0] = normalize(eigenvector0);
    rotation[1] = normalize(eigenvector1);

    translation = mean + p01;

    return true;
}

// from shadertoy: https://www.shadertoy.com/view/stfSzS
float4 BezierAABB(float2 p01, float2 p11, float2 p21)
{
    float2 p0 = p01;
    float2 p1 = p11;
    float2 p2 = p21;

    float2 mi = min(p0, p2);
    float2 ma = max(p0, p2);

    float2 a = p0 - 2.0 * p1 + p2;
    float2 b = p1 - p0;
    float2 t = -b / a; // solution for linear equation at + b = 0

    if (t.x > 0.0 && t.x < 1.0) // x-coord
    {
        float q = QuadraticBezier1D(p0.x, p1.x, p2.x, t.x);

        mi.x = min(mi.x, q);
        ma.x = max(ma.x, q);
    }

    if (t.y > 0.0 && t.y < 1.0) // y-coord
    {
        float q = QuadraticBezier1D(p0.y, p1.y, p2.y, t.y);

        mi.y = min(mi.y, q);
        ma.y = max(ma.y, q);
    }

    return float4(mi, ma);
}

// from shadertoy: https://www.shadertoy.com/view/stfSzS
bool BezierOBB_PCA(float2 p0, float2 p1, float2 p2, float screenSpaceLineWidth, out float2 obbV0, out float2 obbV1, out float2 obbV2, out float2 obbV3)
{
    // try to find transformation of OBB via principal-component-analysis
    float2x2 rotation;
    float2 translation;

    if (estimateTransformation(p0, p1, p2, translation, rotation) == false)
        return false;
            
    // transform Bezier's control-points into the local-space of the OBB
    //
    // 1) instead of using inverse of rot-matrix we can just use transpose of rot-matrix
    //    because rot-matrix is "orthonormal" (each column has unit length and is perpendicular
    //    to every other column)
    // 
    // 2) resulting vector of [transpose(rot) * v] is same as [v * rot] !!!
    
    // compute AABB of curve in local-space
    float4 aabb = BezierAABB(mul(rotation, p0 - translation), mul(rotation, p1 - translation), mul(rotation, p2 - translation));
    aabb.xy -= screenSpaceLineWidth;
    aabb.zw += screenSpaceLineWidth;
    
    // transform AABB back to world-space
    // TODO: Look into better tranforming the aabb back. this computations seem unnecessary
    float2 center = translation + mul((aabb.xy + aabb.zw) / 2.0f, rotation);
    float2 extent = ((aabb.zw - aabb.xy) / 2.0f).xy;
    obbV0 = float2(center + mul(extent, rotation));
    obbV1 = float2(center + mul(float2(extent.x, -extent.y), rotation));
    obbV2 = float2(center + mul(-extent, rotation));
    obbV3 = float2(center + mul(-float2(extent.x, -extent.y), rotation));

    return true;
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

double2 transformPointNdc(ClipProjectionData clipProjectionData, double2 point2d)
{
    float64_t3x3 transformation = clipProjectionData.projectionToNDC;
    return mul(transformation, float64_t3(point2d, 1)).xy;
}
double2 transformVectorNdc(ClipProjectionData clipProjectionData, double2 vector3d)
{
    float64_t3x3 transformation = clipProjectionData.projectionToNDC;
    return mul(transformation, float64_t3(vector3d, 0)).xy;
}
float2 transformPointScreenSpace(ClipProjectionData clipProjectionData, double2 point2d) 
{
    double2 ndc = transformPointNdc(clipProjectionData, point2d);
    return (float2)((ndc + 1.0) * 0.5 * globals.resolution);
}

void BezierOBB_Aligned(float2 p0, float2 p1, float2 p2, float screenSpaceLineWidth, out float2 obbV0, out float2 obbV1, out float2 obbV2, out float2 obbV3)
{
    // shift curve so 'p0' is at origin (will become zero)
    float32_t2 transformedP0 = float32_t2(0.0f, 0.0f);
    float32_t2 transformedP1 = p1 - p0;
    float32_t2 transformedP2 = p2 - p0;
    
    // rotate it around origin so 'p2' is on x-axis
    // 
    // - columns of matrix represents axes of transformed system and we already have one:
    //   normalized vector from origin to p2 represents x-axis of wanted rotated bounding-box
    // - 2nd axis is perpendicular to the 1st one so we just rotate 1st one counter-clockwise
    //   by 90 degrees
    
    const float32_t p2Length = length(transformedP2);
    const float32_t2 axis = transformedP2 / p2Length; // normalized (unit length)
    const float32_t2 translation = p0;
    float32_t2x2 rotation;
    
    rotation[0] = float32_t2(  axis.x, axis.y );      // column 0 ... x-axis
    rotation[1] = float32_t2( -axis.y, axis.x );      // column 1 ... y-axis ... CCW x-axis by 90 degrees
    
    // notes:
    // - rotating 'p0' is pointless as it is "zero" and none rotation will change that
    // - rotating 'p2' will move it to "global" x-axis so its y-coord will be zero and x-coord
    //   will be its distance from origin
    
//  transformed.p0 = transformed.p0 * rotation;
//  transformed.p1 = transformed.p1 * rotation;
//  transformed.p2 = transformed.p2 * rotation;
    
    transformedP1 = mul(rotation, transformedP1);
    transformedP2 = float32_t2(p2Length, 0.0);
    
    // compute AABB of curve in local-space
    float32_t4 aabb = BezierAABB(transformedP0, transformedP1, transformedP2);
    aabb.xy -= screenSpaceLineWidth;
    aabb.zw += screenSpaceLineWidth;
    
    // transform AABB back to world-space
    float32_t2 center = translation + mul((aabb.xy + aabb.zw) / 2.0f, rotation);
    float32_t2 extent = ((aabb.zw - aabb.xy) / 2.0f).xy;
    //float32_t center = p0 + rotation * aabb.center;
    //float32_t2 extent = aabb.extent;
    
    obbV0 = float32_t2(center + mul(extent, rotation));
    obbV1 = float32_t2(center + mul(float32_t2(extent.x, -extent.y), rotation));
    obbV2 = float32_t2(center + mul(-extent, rotation));
    obbV3 = float32_t2(center + mul(-float32_t2(extent.x, -extent.y), rotation));
}

PSInput main(uint vertexID : SV_VertexID)
{
    const uint vertexIdx = vertexID & 0x3u;
    const uint objectID = vertexID >> 2;

    DrawObject drawObj = drawObjects[objectID];

    ObjectType objType = (ObjectType)(((uint32_t)drawObj.type_subsectionIdx) & 0x0000FFFF);
    uint32_t subsectionIdx = (((uint32_t)drawObj.type_subsectionIdx) >> 16);
    PSInput outV;

    // Default Initialize PS Input
    outV.position.z = 0.0;
    outV.data0 = float4(0, 0, 0, 0);
    outV.data1 = uint4(0, 0, 0, 0);
    outV.data2 = float4(0, 0, 0, 0);
    outV.data3 = float4(0, 0, 0, 0);
    outV.clip = float4(0,0,0,0);

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
        outV.setColor(lineStyle.color);
        outV.setLineThickness(screenSpaceLineWidth / 2.0f);

        double2 points[2u];
        points[0u] = vk::RawBufferLoad<double2>(drawObj.geometryAddress, 8u);
        points[1u] = vk::RawBufferLoad<double2>(drawObj.geometryAddress + sizeof(double2), 8u);

        float2 transformedPoints[2u];
        for (uint i = 0u; i < 2u; ++i)
        {
            transformedPoints[i] = transformPointScreenSpace(clipProjectionData, points[i]);
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

        // convert back to ndc
        outV.position.xy = (outV.position.xy / globals.resolution) * 2.0 - 1.0; // back to NDC for SV_Position
        outV.position.w = 1u;
    }
    else if (objType == ObjectType::QUAD_BEZIER)
    {
        outV.setColor(lineStyle.color);
        outV.setLineThickness(screenSpaceLineWidth / 2.0f);

        double2 points[3u];
        points[0u] = vk::RawBufferLoad<double2>(drawObj.geometryAddress, 8u);
        points[1u] = vk::RawBufferLoad<double2>(drawObj.geometryAddress + sizeof(double2), 8u);
        points[2u] = vk::RawBufferLoad<double2>(drawObj.geometryAddress + sizeof(double2) * 2u, 8u);

        // transform these points into screen space and pass to fragment
        float2 transformedPoints[3u];
        for (uint i = 0u; i < 3u; ++i)
        {
            transformedPoints[i] = transformPointScreenSpace(clipProjectionData, points[i]);
        }
        
        nbl::hlsl::shapes::QuadraticBezier<float> quadraticBezier = nbl::hlsl::shapes::QuadraticBezier<float>::construct(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u]);
        nbl::hlsl::shapes::Quadratic<float> quadratic = nbl::hlsl::shapes::Quadratic<float>::constructFromBezier(quadraticBezier);
        nbl::hlsl::shapes::Quadratic<float>::ArcLenCalculator preCompData = nbl::hlsl::shapes::Quadratic<float>::ArcLenCalculator::construct(quadratic);

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
            BezierOBB_Aligned(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], screenSpaceLineWidth / 2.0f, obbV0, obbV1, obbV2, obbV3);
            //BezierOBB_PCA(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], screenSpaceLineWidth / 2.0f, obbV0, obbV1, obbV2, obbV3);
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
            float2 exterior0 = LineLineIntersection(middleExteriorPoint, leftExteriorPoint, midTangent, leftTangent);;
            
            float2 rightTangent = normalize(BezierTangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 1.0f-optimalT));
            float2 rightNormal = normalize(float2(-rightTangent.y, rightTangent.x)) * flip;
            float2 rightExteriorPoint = QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 1.0f-optimalT) - rightNormal * screenSpaceLineWidth / 2.0f;
            float2 exterior1 = LineLineIntersection(middleExteriorPoint, rightExteriorPoint, midTangent, rightTangent);

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
                    outV.position = float4(LineLineIntersection(leftExteriorPoint, endPointExterior, leftTangent, endPointNormal), 0.0, 1.0f);
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
                    outV.position = float4(LineLineIntersection(rightExteriorPoint, endPointExterior, rightTangent, endPointNormal), 0.0, 1.0f);
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
        outV.setColor(lineStyle.color);
        outV.setLineThickness(screenSpaceLineWidth / 2.0f);

        CurveBox curveBox;
        curveBox.aabbMin = vk::RawBufferLoad<double2>(drawObj.geometryAddress, 8u);
        curveBox.aabbMax = vk::RawBufferLoad<double2>(drawObj.geometryAddress + sizeof(double2), 8u);
        for (uint32_t i = 0; i < 3; i ++)
        {
            curveBox.curveMin[i] = vk::RawBufferLoad<double2>(drawObj.geometryAddress + sizeof(double2) * (2 + i), 8u);
            curveBox.curveMax[i] = vk::RawBufferLoad<double2>(drawObj.geometryAddress + sizeof(double2) * (2 + 3 + i), 8u);
        }

        //const double2 ndcAabbExtents = abs(transformVectorNdc(curveBox.aabbMax - curveBox.aabbMin));
        const double2 ndcAabbExtents = double2(
            length(abs(transformVectorNdc(clipProjectionData, double2(curveBox.aabbMax.x, curveBox.aabbMin.y) - curveBox.aabbMin))),
            length(abs(transformVectorNdc(clipProjectionData, double2(curveBox.aabbMin.x, curveBox.aabbMax.y) - curveBox.aabbMin)))
        );
        const double2 dilatedAabbExtents = ndcAabbExtents + 2.0 * (globals.antiAliasingFactor / double2(globals.resolution));
        double2 maxCorner = double2(bool2(vertexIdx & 0x1u, vertexIdx >> 1));
        maxCorner = ((((maxCorner - 0.5) * 2.0 * dilatedAabbExtents) / ndcAabbExtents) + 1.0) * 0.5;
        const double2 coord = transformPointNdc(clipProjectionData, lerp(curveBox.aabbMin, curveBox.aabbMax, maxCorner));
        outV.position = float4((float2) coord, 0.f, 1.f);

        const uint major = (uint)globals.majorAxis;
        const uint minor = 1-major;

        nbl::hlsl::shapes::Quadratic<double> curveMin = nbl::hlsl::shapes::Quadratic<double>::construct(
            curveBox.curveMin[0], curveBox.curveMin[1], curveBox.curveMin[2]);
        nbl::hlsl::shapes::Quadratic<double> curveMax = nbl::hlsl::shapes::Quadratic<double>::construct(
            curveBox.curveMax[0], curveBox.curveMax[1], curveBox.curveMax[2]);

        outV.setMinorBboxUv(maxCorner[minor]);
        outV.setMajorBboxUv(maxCorner[major]);

        nbl::hlsl::equations::Quadratic<float> curveMinMinorAxis = nbl::hlsl::equations::Quadratic<float>::construct(
            (float)curveMin.A[minor], 
            (float)curveMin.B[minor], 
            (float)curveMin.C[minor]);
        nbl::hlsl::equations::Quadratic<float> curveMaxMinorAxis = nbl::hlsl::equations::Quadratic<float>::construct(
            (float)curveMax.A[minor], 
            (float)curveMax.B[minor], 
            (float)curveMax.C[minor]);
        outV.setCurveMinBezier(curveMinMinorAxis);
        outV.setCurveMaxBezier(curveMaxMinorAxis);
        
        nbl::hlsl::equations::Quadratic<float> curveMinRootFinding = nbl::hlsl::equations::Quadratic<float>::construct(
            (float)curveMin.A[major], 
            (float)curveMin.B[major], 
            (float)curveMin.C[major] - maxCorner[major]);
        nbl::hlsl::equations::Quadratic<float> curveMaxRootFinding = nbl::hlsl::equations::Quadratic<float>::construct(
            (float)curveMax.A[major], 
            (float)curveMax.B[major], 
            (float)curveMax.C[major] - maxCorner[major]);
        outV.setMinCurvePrecomputedRootFinders(nbl::hlsl::equations::Quadratic<float>::PrecomputedRootFinder::construct(curveMinRootFinding));
        outV.setMaxCurvePrecomputedRootFinders(nbl::hlsl::equations::Quadratic<float>::PrecomputedRootFinder::construct(curveMaxRootFinding));
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

    outV.clip = float4(outV.position.x - clipProjectionData.minClipNDC.x, outV.position.y - clipProjectionData.minClipNDC.y, clipProjectionData.maxClipNDC.x - outV.position.x, clipProjectionData.maxClipNDC.y - outV.position.y);
    return outV;
}