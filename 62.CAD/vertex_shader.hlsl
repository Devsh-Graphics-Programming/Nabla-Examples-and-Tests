#pragma shader_stage(vertex)

#include "common.hlsl"

PSInput main(uint vertexID : SV_VertexID)
{
    const uint vertexIdx = vertexID & 0x3u;
    const uint objectID = vertexID >> 2;
    
    DrawObject drawObj = drawObjects[objectID];
    ObjectType objType = drawObj.type;
    
    PSInput outV;
    outV.lineWidth_eccentricity_objType.x = asuint(globals.lineWidth);
    outV.lineWidth_eccentricity_objType.z = (uint)objType;

    // TODO: get from styles
    outV.color = globals.color;

    const float antiAliasedLineWidth = globals.lineWidth + globals.antiAliasingFactor * 2.0f;

    if (objType == ObjectType::ELLIPSE)
    {
        outV.color = float4(1.0, 0.0, 0.0, 0.5);
#ifdef LOAD_STRUCT
        EllipseInfo ellipse = vk::RawBufferLoad<EllipseInfo>(drawObj.address, 8u);
        double2 majorAxis = ellipse.majorAxis;
        double2 center = ellipse.majorAxis;
#else
        double2 majorAxis = vk::RawBufferLoad<double2>(drawObj.address, 8u);
        double2 center = vk::RawBufferLoad<double2>(drawObj.address + 16u, 8u);
#endif 
        uint4 angleBoundsPacked_eccentricityPacked_pad = vk::RawBufferLoad<uint4>(drawObj.address + 32u, 8u);

        outV.lineWidth_eccentricity_objType.y = angleBoundsPacked_eccentricityPacked_pad.z; // asfloat because it is acrually packed into a uint and we should not treat it as a float yet.

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
        
        double majorAxisLength = length(transformedMajorAxis);
        double minorAxisLength = majorAxisLength * eccentricity;
        double2 ab = double2(majorAxisLength, -minorAxisLength);
        double2 dir = transformedMajorAxis / majorAxisLength;
        double2 start = ab * double2(cos(angleBounds.x), sin(angleBounds.x));
        double2 end = ab * double2(cos(angleBounds.y), sin(angleBounds.y));
        
        if (vertexIdx == 0u || vertexIdx == 1u)
        {
            outV.position.xy = start + normalize(start) * ((float)vertexIdx - 0.5f) * antiAliasedLineWidth * 10;
        }
        else // if (vertexIdx == 2u || vertexIdx == 3u)
        {
            outV.position.xy = end + normalize(end) * ((float)vertexIdx - 2.5f) * antiAliasedLineWidth * 10;
        }

        // outV.position.xy = mul(double2x2(dir.x, dir.y, -dir.y, dir.x), outV.position.xy);
        outV.position.xy += transformedCenter;
        
        outV.position.xy = (outV.position.xy / globals.resolution) * 2.0 - 1.0; // back to NDC for SV_Position
        outV.position.w = 1u;

        //if (vertexIdx == 0u)
        //    outV.position = float4(-1, -1, 0, 1);
        //else if (vertexIdx == 1u)
        //    outV.position = float4(-1, +1, 0, 1);
        //else if (vertexIdx == 2u)
        //    outV.position = float4(+1, -1, 0, 1);
        //else if (vertexIdx == 3u)
        //    outV.position = float4(+1, +1, 0, 1);
    }
    else if (objType == ObjectType::LINE)
    {
        outV.color = float4(0.0, 0.0, 1.0, 0.5);
        double3x3 transformation = (double3x3)globals.viewProjection;
        LinePoints points = vk::RawBufferLoad<LinePoints>(drawObj.address, 8u);
        float2 transformedPoints[4u];
        for(uint i = 0u; i < 4u; ++i)
        {
            double2 ndc = mul(transformation, double3(points.p[i], 1)).xy; // Transform to NDC
            transformedPoints[i] = (float2)((ndc + 1.0) * 0.5 * globals.resolution); // Transform to Screen Space
        }

        const float2 lineVector = normalize(transformedPoints[2u] - transformedPoints[1u]);
        const float2 normalToLine = float2(-lineVector.y, lineVector.x);

        if (vertexIdx == 0u || vertexIdx == 1u)
        {
            // work in screen space coordinates because of fixed pixel size
            const float2 vectorPrev = normalize(transformedPoints[1u] - transformedPoints[0u]);
            const float2 normalPrevLine = float2(-vectorPrev.y, vectorPrev.x);
            const float2 miter = normalize(normalPrevLine + normalToLine);

            outV.position.xy = transformedPoints[1u] + (miter * ((float)vertexIdx - 0.5f) * (antiAliasedLineWidth)) / dot(normalToLine, miter);
        }
        else // if (vertexIdx == 2u || vertexIdx == 3u)
        {
            // work in screen space coordinates because of fixed pixel size
            const float2 vectorNext = normalize(transformedPoints[3u] - transformedPoints[2u]);
            const float2 normalNextLine = float2(-vectorNext.y, vectorNext.x);
            const float2 miter = normalize(normalNextLine + normalToLine);

            outV.position.xy = transformedPoints[2u] + (miter * ((float)vertexIdx - 2.5f) * (antiAliasedLineWidth)) / dot(normalToLine, miter);
        }

        outV.start_end.xy = transformedPoints[1u];
        outV.start_end.zw = transformedPoints[2u];

        // convert back to ndc
        outV.position.xy = (outV.position.xy / globals.resolution) * 2.0 - 1.0; // back to NDC for SV_Position
        outV.position.w = 1u;
    }
	return outV;
}
