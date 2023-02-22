#pragma shader_stage(vertex)

#include "common.hlsl"

float2 intersectLines2D(in float2 p1, in float2 v1, in float2 v2) /* p2 is zero */
{
    float det = v1.y * v2.x - v1.x * v2.y;
    float2x2 inv = float2x2(v2.y, -v2.x, v1.y, -v1.x) / det;
    float2 t = mul(inv, p1);
    return mul(v2, t.y);
}

PSInput main(uint vertexID : SV_VertexID)
{
    const uint vertexIdx = vertexID & 0x3u;
    const uint objectID = vertexID >> 2;

    DrawObject drawObj = drawObjects[objectID];
    ObjectType objType = drawObj.type;
    
    PSInput outV;
    outV.lineWidth_eccentricity_objType_writeToAlpha.x = asuint(globals.lineWidth);
    outV.lineWidth_eccentricity_objType_writeToAlpha.z = (uint)objType;

    // TODO: get from styles
    outV.color = globals.color;

    const float antiAliasedLineWidth = globals.lineWidth + globals.antiAliasingFactor * 2.0f;

    outV.lineWidth_eccentricity_objType_writeToAlpha.w = (vertexIdx % 2u == 0u) ? 1u : 0u;

    if (objType == ObjectType::ELLIPSE)
    {
        outV.color = float4(0.7, 0.3, 0.1, 0.5);

#ifdef LOAD_STRUCT
        EllipseInfo ellipse = vk::RawBufferLoad<EllipseInfo>(drawObj.address, 8u);
        double2 majorAxis = ellipse.majorAxis;
        double2 center = ellipse.majorAxis;
#else
        double2 majorAxis = vk::RawBufferLoad<double2>(drawObj.address, 8u);
        double2 center = vk::RawBufferLoad<double2>(drawObj.address + 16u, 8u);
#endif 
        uint4 angleBoundsPacked_eccentricityPacked_pad = vk::RawBufferLoad<uint4>(drawObj.address + 32u, 8u);

        outV.lineWidth_eccentricity_objType_writeToAlpha.y = angleBoundsPacked_eccentricityPacked_pad.z; // asfloat because it is acrually packed into a uint and we should not treat it as a float yet.

        double3x3 transformation = (double3x3)globals.viewProjection;

        double2 ndcCenter = mul(transformation, double3(center, 1)).xy; // Transform to NDC
        float2 transformedCenter = (float2)((ndcCenter + 1.0) * 0.5 * globals.resolution); // Transform to Screen Space
        outV.start_end.xy = transformedCenter;

        double2 ndcMajorAxis = mul(transformation, double3(majorAxis, 0)).xy; // Transform to NDC
        float2 transformedMajorAxis = (float2)((ndcMajorAxis) * 0.5 * globals.resolution); // Transform to Screen Space
        outV.start_end.zw = transformedMajorAxis;

        // construct a cage, working in ellipse screen space :
        double2 angleBounds = ((double2)(angleBoundsPacked_eccentricityPacked_pad.xy) / UINT32_MAX) * (2.0 * nbl_hlsl_PI);
        const double eccentricity = (double)(angleBoundsPacked_eccentricityPacked_pad.z) / UINT32_MAX;
        
        float majorAxisLength = length(transformedMajorAxis);
        float minorAxisLength = float(majorAxisLength * eccentricity);
        float2 ab = float2(majorAxisLength, minorAxisLength);
        float2 start = float2(ab * double2(cos(angleBounds.x), sin(angleBounds.x)));
        float2 end = float2(ab * double2(cos(angleBounds.y), sin(angleBounds.y)));
        float2 startToEnd = end - start;

        if (vertexIdx == 0u)
        {
            outV.position.xy = start - normalize(start) * antiAliasedLineWidth * 0.5f;
        }
        else if (vertexIdx == 1u)
        {
            // from p in the direction of startToEnd is the line tangent to ellipse
            float theta = atan2((eccentricity * startToEnd.x), -startToEnd.y) + nbl_hlsl_PI;
            float2 perp = normalize(float2(startToEnd.y, -startToEnd.x));
            float2 p = float2(ab * double2(cos(theta), sin(theta)));
            float2 intersection = intersectLines2D(p + perp * antiAliasedLineWidth * 0.5f, startToEnd, start);
            outV.position.xy = intersection;
        }
        else if (vertexIdx == 2u)
        {
            outV.position.xy = end - normalize(end) * antiAliasedLineWidth * 0.5f;
        }
        else
        {
            // from p in the direction of startToEnd is the line tangent to ellipse
            float theta = atan2((eccentricity * startToEnd.x), -startToEnd.y) + nbl_hlsl_PI;
            float2 perp = normalize(float2(startToEnd.y, -startToEnd.x));
            float2 p = float2(ab * double2(cos(theta), sin(theta)));
            float2 intersection = intersectLines2D(p + perp * antiAliasedLineWidth * 0.5f, startToEnd, end);
            outV.position.xy = intersection;
        }

        // Transform from ellipse screen space to actual screen space
        float2 dir = normalize(transformedMajorAxis);
        outV.position.xy = mul(float2x2(dir.x, dir.y, dir.y, -dir.x), outV.position.xy);
        outV.position.xy += transformedCenter;
        
        // Transform to ndc
        outV.position.xy = (outV.position.xy / globals.resolution) * 2.0 - 1.0; // back to NDC for SV_Position
        outV.position.w = 1u;
    }
    else if (objType == ObjectType::LINE)
    {
        outV.color = float4(0.3, 0.2, 0.6, 0.5);
        double3x3 transformation = (double3x3)globals.viewProjection;

        double2 points[2u];
        points[0u] = vk::RawBufferLoad<double2>(drawObj.address, 8u);
        points[1u] = vk::RawBufferLoad<double2>(drawObj.address + sizeof(double2), 8u);

        float2 transformedPoints[2u];
        for(uint i = 0u; i < 2u; ++i)
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

        outV.start_end.xy = transformedPoints[0u];
        outV.start_end.zw = transformedPoints[1u];

        // convert back to ndc
        outV.position.xy = (outV.position.xy / globals.resolution) * 2.0 - 1.0; // back to NDC for SV_Position
        outV.position.w = 1u;
    }
    else if (objType == ObjectType::ROAD)
    {
        outV.color = float4(0.7, 0.1, 0.5, 0.5);
        double3x3 transformation = (double3x3)globals.viewProjection;
        double2 vertex = vk::RawBufferLoad<double2>(drawObj.address + sizeof(double2) * vertexIdx, 8u);
        outV.position.xy = mul(transformation, double3(vertex, 1)).xy; // Transform to NDC
        outV.position.w = 1u;
    }
	return outV;
}
