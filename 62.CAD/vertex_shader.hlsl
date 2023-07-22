#pragma shader_stage(vertex)

#include "common.hlsl"

float2 Tangent(float2 p0, float2 p1, float2 p2, float t)
{
    float2 tangent = 2.0 * (1.0 - t) * (p1 - p0) + 2.0 * t * (p2 - p1);
    return tangent;
}

float TforTan(float2 p0, float2 p1, float2 p2, float2 DirNormalized)
{
    float a = -2 * (p1.x - p0.x) + 2 * (p2.x - p1.x);
        float b = 2 * (p1.x - p0.x);
        float c = -2 * (p1.y - p0.y) + 2 * (p2.y - p1.y);
        float d = 2 * (p1.y - p0.y);
        float e = DirNormalized.x;
        float f = DirNormalized.y;

        float num = float(d * e - b * f);
        float den = float(a * f - c * e);
        float t;
    if (den == 0) return 0;
    else t = num / den;
    return t;
}

float2 QuadraticBezier(float2 p0, float2 p1, float2 p2, float t)
{
    float tt = t * t;
    float oneMinusT = 1.0 - t;
    float oneMinusTT = oneMinusT * oneMinusT;
    float2 position = p0 * oneMinusTT + 2.0 * p1 * oneMinusT * t + p2 * tt;
    return position;
}

float2 CalculateQuadraticBezierExtrema(float2 P0, float2 P1, float2 P2)
{
    float2 extrema;

    // Calculate the coefficients of the quadratic equation
    float2 A = P0 - 2.0 * P1 + P2;
    float2 B = 2.0 * (P1 - P0);

    // Calculate the x-coordinate of the extrema
    if (A.x != 0.0)
    {
        float discriminantX = B.x * B.x - 4.0 * A.x * A.y;
        if (discriminantX >= 0.0)
        {
            float t = (-B.x + sqrt(discriminantX)) / (2.0 * A.x);
            if (t >= 0.0 && t <= 1.0)
                extrema.x = P0.x * (1.0 - t) * (1.0 - t) + 2.0 * P1.x * (1.0 - t) * t + P2.x * t * t;
        }
    }

    // Calculate the y-coordinate of the extrema
    if (A.y != 0.0)
    {
        float discriminantY = B.y * B.y - 4.0 * A.y * A.x;
        if (discriminantY >= 0.0)
        {
            float t = (-B.y + sqrt(discriminantY)) / (2.0 * A.y);
            if (t >= 0.0 && t <= 1.0)
                extrema.y = P0.y * (1.0 - t) * (1.0 - t) + 2.0 * P1.y * (1.0 - t) * t + P2.y * t * t;
        }
    }

    return extrema;
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

float3 normalize(float3 A) {
    return A / sqrt(dot(A, A));
}

float BezierCurve(float2 P0, float2 P1, float2 P2, float T)
{
    // Calculate the tangent vectors at the start and end points
    float2 tangent0 = P1 - P0;
    float2 tangent1 = P2 - P1;

    // Calculate the point on the curve at parameter T
    float2 Q = (1.0 - T) * ((1.0 - T) * P0 + T * P1) + T * ((1.0 - T) * P1 + T * P2);

    // Calculate the derivative of the curve at parameter T
    float2 dQdT = 2.0 * (1.0 - T) * (P1 - P0) + 2.0 * T * (P2 - P1);

    // Calculate the second derivative of the curve at parameter T (curvature)
    float2 ddQdT2 = 2.0 * (P2 - 2.0 * P1 + P0);

    // Calculate the curvature using the formula: curvature = |dQ/dT| / |dQ/dT|^3
    float curvature = length(ddQdT2) / pow(length(dQdT), 3.0);

    return curvature;
}

PSInput main(uint vertexID : SV_VertexID)
{
    const uint vertexIdx = vertexID & 0x3u;
    const uint objectID = vertexID >> 2;

    DrawObject drawObj = drawObjects[objectID];
    ObjectType objType = (ObjectType)(((uint32_t)drawObj.type) & 0x0000FFFF);
    uint32_t SubobjectIdx = (((uint32_t)drawObj.type) >> 16);
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

        float2 Center = QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.5f);
        float2 CenterT = normalize(Tangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.5f));
        float2 CenterN = normalize(float2(-CenterT.y, CenterT.x));
        float2 tangent;
        float2 normal;
        float2 CenterSigns = dot(normalize(transformedPoints[1u] - (transformedPoints[0u] + transformedPoints[2u]) / 2.0f), CenterN) >= 0 ? 1 : -1;
        CenterN *= -CenterSigns;

        //     float2 Diff1 = (QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.26f) - QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.25f)) / -0.1f;
            // float2 p0 = QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.25f) - normalize(float2(-Tangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.25f).y, Tangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.25f).x)) * -float2(-Diff1.y, Diff1.x);
             //loat2 p1 = QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.75f) - normalize(float2(-Tangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.75f).y, Tangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.75f).x)) * screenSpaceLineWidth / 1.5f;


        float2 p0 = (transformedPoints[0u] + transformedPoints[1u]) / 2.0f;
        float2 p1 = (transformedPoints[1u] + transformedPoints[2u]) / 2.0f;


#if 0
        float2 Line1V1 = Center - CenterN * screenSpaceLineWidth / 2.0f + CenterT * 1000.0f;
        float2 Line1V2 = Center - CenterN * screenSpaceLineWidth / 2.0f - CenterT * 1000.0f;
        {
            float2 Line2V1 = p0 + normalize(transformedPoints[1u] - transformedPoints[0u]) * 1000.0f;
            float2 Line2V2 = p0 - normalize(transformedPoints[1u] - transformedPoints[0u]) * 1000.0f;

            p0 = CalculateIntersection(Line1V1, Line2V1, Line1V2, Line2V2);
        }
        {
            float2 Line2V1 = p1 + normalize(transformedPoints[1u] - transformedPoints[2u]) * 1000.0f;
            float2 Line2V2 = p1 - normalize(transformedPoints[1u] - transformedPoints[2u]) * 1000.0f;

            p1 = CalculateIntersection(Line1V1, Line2V1, Line1V2, Line2V2);
        }


        switch (SubobjectIdx) {
        case 0:

            tangent = normalize(Tangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.0f));
            normal = normalize(float2(-tangent.y, tangent.x));
            normal *= dot(normal, CenterN) < 0 ? -1 : 1;

            if (vertexIdx == 0u)
                outV.position = float4(transformedPoints[0u] - normal * screenSpaceLineWidth - tangent * screenSpaceLineWidth / 2.0f, 0.0, 1.0f);
            else if (vertexIdx == 1u)
                outV.position = float4(transformedPoints[0u] + normal * screenSpaceLineWidth / 2.0f - tangent * screenSpaceLineWidth / 2.0f, 0.0, 1.0f);
            else if (vertexIdx == 2u)
                outV.position = float4(p0, 0.0, 1.0f);
            else if (vertexIdx == 3u)
                outV.position = float4(Center + CenterN * screenSpaceLineWidth / 2.0f, 0.0, 1.0f);
            break;
        case 1:
            if (vertexIdx == 0u)
                outV.position = float4(p0, 0.0, 1.0f);
            else if (vertexIdx == 1u)
                outV.position = float4(p0, 0.0, 1.0f);
            else if (vertexIdx == 2u)
                outV.position = float4(Center + CenterN * screenSpaceLineWidth / 2.0f, 0.0, 1.0f);
            else if (vertexIdx == 3u)
                outV.position = float4(p1, 0.0, 1.0f);
            break;
        case 2:
            tangent = normalize(Tangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 1.0f));
            normal = normalize(float2(-tangent.y, tangent.x));
            normal *= dot(normal, CenterN) < 0 ? -1 : 1;
            if (vertexIdx == 0u)
                outV.position = float4(transformedPoints[2u] - normal * screenSpaceLineWidth + tangent * screenSpaceLineWidth / 2.0f, 0.0, 1.0f);
            else if (vertexIdx == 1u)
                outV.position = float4(transformedPoints[2u] + normal * screenSpaceLineWidth / 2.0f + tangent * screenSpaceLineWidth / 2.0f, 0.0, 1.0f);
            else if (vertexIdx == 2u)
                outV.position = float4(p1, 0.0, 1.0f);
            else if (vertexIdx == 3u)
                outV.position = float4(Center + CenterN * screenSpaceLineWidth / 2.0f, 0.0, 1.0f);
            break;
        }
#endif
#if 1


        float2 Line1V1 = Center - CenterN * screenSpaceLineWidth / 2.0f + CenterT * 1000.0f;
        float2 Line1V2 = Center - CenterN * screenSpaceLineWidth / 2.0f - CenterT * 1000.0f;
        float2 Line2V1;
        float2 Line2V2;
        float2 Line3V1;
        float2 Line3V2;
        {
            float2 BaseLine = (normalize(Center - transformedPoints[0u]) + normalize(transformedPoints[1u] - transformedPoints[0u])) / 2.0f;
            float T = saturate(TforTan(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], BaseLine));
            tangent = normalize(Tangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], T));
            normal = normalize(float2(-tangent.y, tangent.x));
            normal *= dot(normal, CenterN) < 0 ? -1 : 1;

            Line2V1 = QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], T) + tangent * 1000.0f - normal * screenSpaceLineWidth / 2.0f;
            Line2V2 = QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], T) - tangent * 1000.0f - normal * screenSpaceLineWidth / 2.0f;

            p0 = CalculateIntersection(Line1V1, Line2V1, Line1V2, Line2V2);
        }
        {
            float2 BaseLine = (normalize(Center - transformedPoints[2u]) + normalize(transformedPoints[1u] - transformedPoints[2u])) / 2.0f;
            float T = saturate(TforTan(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], BaseLine));
            tangent = normalize(Tangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], T));
            normal = normalize(float2(-tangent.y, tangent.x));
            normal *= dot(normal, CenterN) < 0 ? -1 : 1;

            Line3V1 = QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], T) + tangent * 1000.0f - normal * screenSpaceLineWidth / 2.0f;
            Line3V2 = QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], T) - tangent * 1000.0f - normal * screenSpaceLineWidth / 2.0f;

            p1 = CalculateIntersection(Line1V1, Line3V1, Line1V2, Line3V2);
        }

        float2 IP0;
        float2 IP1;

        {
            tangent = normalize(Tangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.35f));
            normal = normalize(float2(-tangent.y, tangent.x));
            normal *= dot(normal, CenterN) < 0 ? -1 : 1;
            IP0 = QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.35f) + normal * screenSpaceLineWidth / 2.0f;
        }

        {
            tangent = normalize(Tangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.65f));
            normal = normalize(float2(-tangent.y, tangent.x));
            normal *= dot(normal, CenterN) < 0 ? -1 : 1;
            IP1 = QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.65f) + normal * screenSpaceLineWidth / 2.0f;
        }
        float2 Line0V1;
        float2 Line0V2;
        switch (SubobjectIdx) {
        case 0:


            tangent = normalize(Tangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.0f));
            normal = normalize(float2(-tangent.y, tangent.x));
            normal *= dot(normal, CenterN) < 0 ? -1 : 1;
            
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
            tangent = normalize(Tangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 1.0f));
            normal = normalize(float2(-tangent.y, tangent.x));
            normal *= dot(normal, CenterN) < 0 ? -1 : 1;

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
#endif

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
    if (vertexIdx == 0u)
        outV.position = float4(-1, -1, 0, 1);
    else if (vertexIdx == 1u)
        outV.position = float4(-1, +1, 0, 1);
    else if (vertexIdx == 2u)
        outV.position = float4(+1, -1, 0, 1);
    else if (vertexIdx == 3u)
        outV.position = float4(+1, +1, 0, 1);
#endif

    return outV;
}