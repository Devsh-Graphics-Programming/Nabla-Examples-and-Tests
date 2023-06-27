#pragma shader_stage(vertex)

#include "common.hlsl"

PSInput main(uint vertexID : SV_VertexID)
{
    const uint vertexIdx = vertexID & 0x3u;
    const uint objectID = vertexID >> 2;

    DrawObject drawObj = drawObjects[objectID];
    ObjectType objType = drawObj.type;
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
        
        // TODO[Payton]:
        // This is where you will be operation and generating the cage based on the bezier points
        // As you can see right now to get away with just drawing the bezier I make the single "cage" fullscreen so we make sure we can see the fragment shadedr work.
        // But to we need to implement tight cages to avoid this huge overdraw.
        // So we will decide the position of the vertex (in ndc space) based on:
            // vertexId + sectionIdx + antiAliasedLineWidth + transformedPoints
        // make sure to work in screen space first (because antiAliasedLineWidth and transformedPoints are in screenSpace) and then transform back to ndc
            // see how we generate a single cage for the line above
        // We will split Quadratic Beziers into 3 sections always.
        // The reason we do this in vertex shader (do calc in screen and transform back to ndc) is because the lineWidth can have fixed size in screenspace
            // imagine zooming in/out but the line width is 2 pixels always
            // the vertices should obviously move outwards in screen when we zoom in, but they shouldn't get bigger so we can't simply transform them by applying a simple projection matrix like we do in graphics 101
            // so vertex positioning is dependant on the projection matrix and we do this here instead of moving vertices each frame on the cpu and then uploading them to gpu
        if (vertexIdx == 0u)
            outV.position = float4(-1, -1, 0, 1);
        else if (vertexIdx == 1u)
            outV.position = float4(-1, +1, 0, 1);
        else if (vertexIdx == 2u)
            outV.position = float4(+1, -1, 0, 1);
        else if (vertexIdx == 3u)
            outV.position = float4(+1, +1, 0, 1);
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