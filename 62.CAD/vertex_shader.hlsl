#pragma shader_stage(vertex)

#include "common.hlsl"

float2 BezierTangent(float2 p0, float2 p1, float2 p2, float t)
{
    return 2.0 * (1.0 - t) * (p1 - p0) + 2.0 * t * (p2 - p1);
}

float2 QuadraticBezier(float2 p0, float2 p1, float2 p2, float t)
{   
    float tt = t * t;
    float oneMinusT = 1.0f - t;
    float oneMinusTT = oneMinusT * oneMinusT;
    return p0 * oneMinusTT + 2.0f * p1 * oneMinusT * t + p2 * tt;
}


float2 CalculateIntersection(float2 line1Start, float2 line2Start, float2 line1End, float2 line2End)
{
    float2 intersectionPoint;

    float2 line1Direction = normalize(line1End - line1Start);
    float2 line2Direction = normalize(line2End - line2Start);

    float line1Denominator = line2Direction.y * line1Direction.x - line2Direction.x * line1Direction.y;

    if (abs(line1Denominator) < 0.0001)
    {
        // Lines are parallel, return a default value or handle accordingly.
        intersectionPoint = float2(0, 0); // Default value, adjust as needed.
    }
    else
    {
        float2 line2StartToLine1Start = line1Start - line2Start;
        float line2Numerator = line2Direction.x * line2StartToLine1Start.y - line2Direction.y * line2StartToLine1Start.x;

        float t = line2Numerator / line1Denominator;
        intersectionPoint = line1Start + t * line1Direction;
    }

    return intersectionPoint;
}


//from shadertoy: https://www.shadertoy.com/view/stfSzS
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
//Compute bezier in one dimension, as the OBB X and Y are at different T's
float bezierEval(float v0, float v1, float v2, float t)
{
    float s = 1.0 - t;

    return v0 * (s * s) +
        v1 * (s * t * 2.0) +
        v2 * (t * t);
}

float4 bezierAABB(float2 p01, float2 p11, float2 p21)
{
    float2 p0 = p01;
    float2 p1 = p11;
    float2 p2 = p21;

    float2 mi = min(p0, p2);
    float2 ma = max(p0, p2);

    float2 a = p0 - 2.0 * p1 + p2;
    float2 b = p1 - p0;
    float2 t = -b / a;             // solution for linear equation at + b = 0    

    if (t.x > 0.0 && t.x < 1.0) // x-coord
    {
        float q = bezierEval(p0.x, p1.x, p2.x, t.x);

        mi.x = min(mi.x, q);
        ma.x = max(ma.x, q);
    }

    if (t.y > 0.0 && t.y < 1.0) // y-coord
    {
        float q = bezierEval(p0.y, p1.y, p2.y, t.y);

        mi.y = min(mi.y, q);
        ma.y = max(ma.y, q);
    }

    return float4(mi, ma);
}

bool bezierOBB_PCA(float2 p0, float2 p1, float2 p2, out float4 Pos0, out float4 Pos1, float screenSpaceLineWidth)
{
    float2x2 rotation;
    float2 translation;

    if (estimateTransformation(p0, p1, p2, translation, rotation) == false)
        return false;

    float4 aabb = bezierAABB(mul(rotation, p0 - translation), mul(rotation, p1 - translation), mul(rotation, p2 - translation));
    aabb.xy -= screenSpaceLineWidth;
    aabb.zw += screenSpaceLineWidth;
    float2 center = translation + mul((aabb.xy + aabb.zw) / 2.0f, rotation);
    float2 Extent = ((aabb.zw - aabb.xy) / 2.0f).xy;
    Pos0 = float4(center + mul(Extent, rotation), center + mul(float2(Extent.x, -Extent.y), rotation));
    Pos1 = float4(center + mul(-Extent, rotation), center + mul(-float2(Extent.x, -Extent.y), rotation));

    return true;
}

PSInput main(uint vertexID : SV_VertexID)
{
    const uint vertexIdx = vertexID & 0x3u;
    const uint objectID = vertexID >> 2;

    DrawObject drawObj = drawObjects[objectID];
    ObjectType objType = (ObjectType)(((uint32_t)drawObj.type_subsectionIdx) & 0x0000FFFF);
    uint32_t subsectionIdx = (((uint32_t)drawObj.type_subsectionIdx) >> 16);
    PSInput outV;

    outV.setObjType(objType);
    outV.setWriteToAlpha((vertexIdx % 2u == 0u) ? 1u : 0u);

    // We only need these for Outline type objects like lines and bezier curves
    LineStyle lineStyle = lineStyles[drawObj.styleIdx];
    const float screenSpaceLineWidth = lineStyle.screenSpaceLineWidth + float(lineStyle.worldSpaceLineWidth * globals.screenToWorldRatio);
    const float antiAliasedLineWidth = screenSpaceLineWidth + globals.antiAliasingFactor * 2.0f;

    if (objType == ObjectType::LINE)
    {
        outV.setColor(lineStyle.color);
        outV.setLineThickness(screenSpaceLineWidth / 2.0f);

        double3x3 transformation = (double3x3)globals.viewProjection;

        double2 points[2u];
        points[0u] = vk::RawBufferLoad<double2>(drawObj.address, 8u);
        points[1u] = vk::RawBufferLoad<double2>(drawObj.address + sizeof(double2), 8u);

        float2 transformedPoints[2u];
        for (uint i = 0u; i < 2u; ++i)
        {
            double2 ndc = mul(transformation, double3(points[i], 1)).xy; // Transform to NDC
            transformedPoints[i] = (float2)((ndc + 1.0) * 0.5 * globals.resolution); // Transform to Screen Space
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

        double3x3 transformation = (double3x3)globals.viewProjection;

        double2 points[3u];
        points[0u] = vk::RawBufferLoad<double2>(drawObj.address, 8u);
        points[1u] = vk::RawBufferLoad<double2>(drawObj.address + sizeof(double2), 8u);
        points[2u] = vk::RawBufferLoad<double2>(drawObj.address + sizeof(double2) * 2u, 8u);

        // transform these points into screen space and pass to fragment
        float2 transformedPoints[3u];
        for (uint i = 0u; i < 3u; ++i)
        {
            double2 ndc = mul(transformation, double3(points[i], 1)).xy; // Transform to NDC
            transformedPoints[i] = (float2)((ndc + 1.0) * 0.5 * globals.resolution); // Transform to Screen Space
        }

        outV.setBezierP0(transformedPoints[0u]);
        outV.setBezierP1(transformedPoints[1u]);
        outV.setBezierP2(transformedPoints[2u]);

        //Whether or not to flip the the interior cage nodes
        int flip = cross(float3(transformedPoints[0u] - transformedPoints[1u], 0), float3(transformedPoints[2u] - transformedPoints[1u], 0)).z > 0 ? -1 : 1;

        float2 Center = QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.5f);
        float2 CenterT = normalize(BezierTangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.5f));
        float2 CenterN = normalize(float2(-CenterT.y, CenterT.x)) * flip;

        //re-used data
        float2 tangent;
        float2 normal;
        //exterior cage points
        float2 p0;
        float2 p1;
        //Internal cage points
        float2 IP0;
        float2 IP1;

        float2 Mid = (transformedPoints[0u] + transformedPoints[2u]) / 2.0f;
        float Radius = length(Mid - transformedPoints[0u]) / 2.0f;
        
        float2 vectorAB = transformedPoints[1u] - transformedPoints[0u];
        float2 vectorAC = transformedPoints[2u] - transformedPoints[1u];

        //T with max curve
        float MaxCurveT = dot(-(transformedPoints[1u] - transformedPoints[0u]), transformedPoints[2u] - 2.0f * transformedPoints[1u] + transformedPoints[0u]) / pow(length(transformedPoints[2u] - 2.0f * transformedPoints[1u] + transformedPoints[0u]),2.0f);

        float area = abs(vectorAB.x * vectorAC.y - vectorAB.y * vectorAC.x) * 0.5;
        float MaxCurve;
        if (length(transformedPoints[1u] - lerp(transformedPoints[0u], transformedPoints[2u], 0.25f)) > Radius && length(transformedPoints[1u] - lerp(transformedPoints[0u], transformedPoints[2u], 0.75f)) > Radius)
            MaxCurve = pow(length(transformedPoints[1u] - Mid), 3) / (area * area);
        else 
            MaxCurve = max(area / pow(length(transformedPoints[0u] - transformedPoints[1u]), 3), area / pow(length(transformedPoints[2u] - transformedPoints[1u]), 3));

        if (MaxCurve * screenSpaceLineWidth > 16) {
            //OBB Fallback
            float4 Pos0;
            float4 Pos1;
            if (subsectionIdx == 0 && bezierOBB_PCA(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], Pos0, Pos1, screenSpaceLineWidth / 2.0f)) {
                if (vertexIdx == 0u)
                    outV.position = float4(Pos0.xy, 0.0, 1.0f);
                else if (vertexIdx == 1u)
                    outV.position = float4(Pos0.zw, 0.0, 1.0f);
                else if (vertexIdx == 2u)
                    outV.position = float4(Pos1.zw, 0.0, 1.0f);
                else if (vertexIdx == 3u)
                    outV.position = float4(Pos1.xy, 0.0, 1.0f);
            }
            else {
                outV.position = float4(0, 0, 0.0, 1.0f);
            }
        } else {
            float2 Line1V1 = Center - CenterN * screenSpaceLineWidth / 2.0f + CenterT * 1000.0f;
            float2 Line1V2 = Center - CenterN * screenSpaceLineWidth / 2.0f - CenterT * 1000.0f;
            float2 Line2V1;
            float2 Line2V2;
            float2 Line3V1;
            float2 Line3V2;
            //External Cage Points
            {
                tangent = normalize(BezierTangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.145f));
                normal = normalize(float2(-tangent.y, tangent.x)) * flip;

                Line2V1 = QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.145f) + tangent * 1000.0f - normal * screenSpaceLineWidth / 2.0f;
                Line2V2 = QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.145f) - tangent * 1000.0f - normal * screenSpaceLineWidth / 2.0f;
                //Calculating intersection between tangent line of the center(Line1) and the tangent line of the left side of bezier
                p0 = CalculateIntersection(Line1V1, Line2V1, Line1V2, Line2V2);
            }
            {
                tangent = normalize(BezierTangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.855f));
                normal = normalize(float2(-tangent.y, tangent.x)) * flip;

                Line3V1 = QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.855f) + tangent * 1000.0f - normal * screenSpaceLineWidth / 2.0f;
                Line3V2 = QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.855f) - tangent * 1000.0f - normal * screenSpaceLineWidth / 2.0f;
                //Calculating intersection between tangent line of the center(Line1) and the tangent line of the right side of bezier
                p1 = CalculateIntersection(Line1V1, Line3V1, Line1V2, Line3V2);
            }

            {
                tangent = normalize(BezierTangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.286));
                normal = normalize(float2(-tangent.y, tangent.x)) * flip;
                IP0 = QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.286) + normal * screenSpaceLineWidth / 2.0f;
            }
            {
                tangent = normalize(BezierTangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.714f));
                normal = normalize(float2(-tangent.y, tangent.x)) * flip;
                IP1 = QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.714f) + normal * screenSpaceLineWidth / 2.0f;
            }
            //Adaptive Inner Cage
            if (MaxCurve * screenSpaceLineWidth / 2.0f > 0.5f) {
                float2 TanMaxCurve = (normalize(BezierTangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], MaxCurveT)));
                float2 MaxCurvePos = QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], MaxCurveT);
                float2 TP0 = transformedPoints[0u] - MaxCurvePos;
                float2 TP1 = transformedPoints[1u] - MaxCurvePos;
                float2 TP2 = transformedPoints[2u] - MaxCurvePos;
                float angle = atan2(TanMaxCurve.y, TanMaxCurve.x);
                float cos_angle = cos(angle);
                float sin_angle = sin(angle);
                float2x2 rotmat = float2x2(cos_angle, sin_angle, -sin_angle, cos_angle);
                TP0 = mul(rotmat, TP0);
                TP1 = mul(rotmat, TP1);
                TP2 = mul(rotmat, TP2);

                float m1 = (TP1.y - TP0.y) / (TP1.x - TP0.x);
                float m2 = (TP2.y - TP1.y) / (TP2.x - TP1.x);

                float a = (m1 - m2) / (-2.0f * TP2.x + 2.0f * TP0.x);

                float y = (a * pow(screenSpaceLineWidth / 2.0f, 2)) + (1.0f / (4.0f * a));
                IP0 = mul(float2(0, y), rotmat) + MaxCurvePos;
                IP1 = IP0;
            }

            float2 Line0V1;
            float2 Line0V2;
            switch (subsectionIdx) {
            case 0:
                tangent = normalize(BezierTangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.0f));
                normal = normalize(float2(-tangent.y, tangent.x)) * flip;

                Line0V1 = transformedPoints[0u] - normal * 1000.0f - tangent * screenSpaceLineWidth / 2.0f;
                Line0V2 = transformedPoints[0u] + normal * 1000.0f - tangent * screenSpaceLineWidth / 2.0f;

                if (vertexIdx == 0u)
                    outV.position = float4(CalculateIntersection(Line2V1, Line0V1, Line2V2, Line0V2), 0.0, 1.0f);
                else if (vertexIdx == 1u)
                    outV.position = float4(transformedPoints[0u] + normal * screenSpaceLineWidth / 2.0f - tangent * screenSpaceLineWidth / 2.0f, 0.0, 1.0f);
                else if (vertexIdx == 2u)
                    outV.position = float4(p0, 0.0, 1.0f);
                else if (vertexIdx == 3u)
                    outV.position = float4(IP0, 0.0, 1.0f);
                break;
            case 1:
                if (vertexIdx == 0u)
                    outV.position = float4(p0, 0.0, 1.0f);
                else if (vertexIdx == 1u)
                    outV.position = float4(IP0, 0.0, 1.0f);
                else if (vertexIdx == 2u)
                    outV.position = float4(p1, 0.0, 1.0f);
                else if (vertexIdx == 3u)
                    outV.position = float4(IP1, 0.0, 1.0f);
                break;
            case 2:
                tangent = normalize(BezierTangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 1.0f));
                normal = normalize(float2(-tangent.y, tangent.x)) * flip;

                Line0V1 = transformedPoints[2u] - normal * 1000.0f + tangent * screenSpaceLineWidth / 2.0f;
                Line0V2 = transformedPoints[2u] + normal * 1000.0f + tangent * screenSpaceLineWidth / 2.0f;

                if (vertexIdx == 0u)
                    outV.position = float4(CalculateIntersection(Line3V1, Line0V1, Line3V2, Line0V2), 0.0, 1.0f);
                else if (vertexIdx == 1u)
                    outV.position = float4(transformedPoints[2u] + normal * screenSpaceLineWidth / 2.0f + tangent * screenSpaceLineWidth / 2.0f, 0.0, 1.0f);
                else if (vertexIdx == 2u)
                    outV.position = float4(p1, 0.0, 1.0f);
                else if (vertexIdx == 3u)
                    outV.position = float4(IP1, 0.0, 1.0f);
            }
        }


        outV.position.xy = (outV.position.xy / globals.resolution) * 2.0 - 1.0;


    }
    /*
        TODO[Lucas]:
        Another `else if` for CurveBox Object Type,
        What you basically need to do here is transform the box min,max and set `outV.position` correctly based on that + vertexIdx
        and you need to do certain outV.setXXX() functions to set the correct (transformed) data to frag shader

        Another note: you may know that for transparency we draw objects twice
        only when `writeToAlpha` is true (even provoking vertex), sdf calculations will happen and alpha will be set
        otherwise it's just a second draw to "Resolve" and the only important thing on "Resolves" is the same `outV.position` as the previous draw (basically the same cage)
        So if you do any precomputation, etc for sdf caluclations you could skip that :D and save yourself the trouble if `writeToAlpha` is false.
    */
    
    
// Make the cage fullscreen for testing:
#if 0
    if (subsectionIdx == 0) {
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

    return outV;
}