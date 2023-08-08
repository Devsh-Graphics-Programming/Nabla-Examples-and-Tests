#pragma shader_stage(fragment)

#include "common.hlsl"
#include <nbl/builtin/hlsl/shapes/rounded_line.hlsl>
#include <nbl/builtin/hlsl/shapes/beziers.hlsl>

#if defined(NBL_FEATURE_FRAGMENT_SHADER_PIXEL_INTERLOCK)
[[vk::ext_instruction(/* OpBeginInvocationInterlockEXT */ 5364)]]
void beginInvocationInterlockEXT();
[[vk::ext_instruction(/* OpEndInvocationInterlockEXT */ 5365)]]
void endInvocationInterlockEXT();
#endif

float4 main(PSInput input) : SV_TARGET
{
#if defined(NBL_FEATURE_FRAGMENT_SHADER_PIXEL_INTERLOCK)
    [[vk::ext_capability(/*FragmentShaderPixelInterlockEXT*/ 5378)]]
    [[vk::ext_extension("SPV_EXT_fragment_shader_interlock")]]
    vk::ext_execution_mode(/*PixelInterlockOrderedEXT*/ 5366);
#endif
	
    // TODO is this going to interact badly with the pixel interlock?	
    if (globals.clipEnabled != 0 && (any(input.position.xy < globals.clip.xy) || any(input.position.xy > globals.clip.zw))) discard;
    
    ObjectType objType = input.getObjType();
    float localAlpha = 0.0f;
    bool writeToAlpha = input.getWriteToAlpha() == 1u;

    // for hatches in the fragment shader we don't need to do the alpha stuff we do to avoid polyline self intersection
    if (objType == ObjectType::CURVE_BOX)
    {
        float2 positionFullscreen = (input.position.xy - 0.5) / float2(globals.resolution);

        nbl::hlsl::shapes::QuadraticBezier curveMin = nbl::hlsl::shapes::QuadraticBezier::construct(
            input.getCurveMinP0(),
            input.getCurveMinP1(),
            input.getCurveMinP2(),
            0.0
        );
        nbl::hlsl::shapes::QuadraticBezier curveMax = nbl::hlsl::shapes::QuadraticBezier::construct(
            input.getCurveMaxP0(),
            input.getCurveMaxP1(),
            input.getCurveMaxP2(),
            0.0
        );
        // TODO: Use flexible major coordinate
        const uint majorCoordinate = 0;
        float minT = curveMin.tForMajorCoordinate(majorCoordinate, positionFullscreen[1 - majorCoordinate]);
        float min = curveMin.evaluate(minT)[majorCoordinate];
        float maxT = curveMax.tForMajorCoordinate(majorCoordinate, positionFullscreen[1 - majorCoordinate]);
        float max = curveMax.evaluate(maxT)[majorCoordinate];
        
        // TODO: anti aliasing
        float4 col = input.getColor();
        float alpha = 0.0;
        if (positionFullscreen[majorCoordinate] >= min && positionFullscreen[majorCoordinate] <= max)
        {
            alpha = 1.0;
        }
        return float4(col.xyz, col.w * alpha);
    }
    else // if (objType != ObjectType::CURVE_BOX)
    {
        if (writeToAlpha)
        {
            if (objType == ObjectType::LINE)
            {
                const float2 start = input.getLineStart();
                const float2 end = input.getLineEnd();
                const float lineThickness = input.getLineThickness();

                float distance = nbl::hlsl::shapes::RoundedLine_t::construct(start, end, lineThickness).signedDistance(input.position.xy);

                const float antiAliasingFactor = globals.antiAliasingFactor;
                localAlpha = 1.0f - smoothstep(-antiAliasingFactor, +antiAliasingFactor, distance);
            }
            else if (objType == ObjectType::QUAD_BEZIER)
            {
                const float2 a = input.getBezierP0();
                const float2 b = input.getBezierP1();
                const float2 c = input.getBezierP2();
                const float lineThickness = input.getLineThickness();

                float distance = nbl::hlsl::shapes::QuadraticBezier::construct(a, b, c, lineThickness).signedDistance(input.position.xy);

                const float antiAliasingFactor = globals.antiAliasingFactor;
                localAlpha = 1.0f - smoothstep(-antiAliasingFactor, +antiAliasingFactor, distance);
            }
            /*
            TODO[Lucas]:
                Another else case for CurveBox where you simply do what I said in the notes of common.hlsl PSInput
                and solve two quadratic equations, you could check for it being a "line" for the mid point being nan
                you will use input.getXXX() to get values needed for this computation
            */
        }

        uint2 fragCoord = uint2(input.position.xy);

        float alpha = 0.0f; // new alpha

#if defined(NBL_FEATURE_FRAGMENT_SHADER_PIXEL_INTERLOCK)
        beginInvocationInterlockEXT();

        alpha = asfloat(pseudoStencil[fragCoord]);
        if (writeToAlpha)
        {
            if (localAlpha > alpha)
                pseudoStencil[fragCoord] = asuint(localAlpha);
        }
        else
        {
            if (alpha != 0.0f)
                pseudoStencil[fragCoord] = asuint(0.0f);
        }

        endInvocationInterlockEXT();
        
        if (writeToAlpha || alpha == 0.0f)
            discard;
#else
        alpha = localAlpha;
        if (!writeToAlpha)
            discard;
        //if (writeToAlpha)
        //{
        //    InterlockedMax(pseudoStencil[fragCoord], asuint(localAlpha));
        //}
        //else
        //{
        //    uint previousAlpha;
        //    InterlockedExchange(pseudoStencil[fragCoord], 0u, previousAlpha);
        //    alpha = previousAlpha;
        //}
#endif

        float4 col = input.getColor();
        return float4(col.xyz, col.w * alpha);
    }
}   