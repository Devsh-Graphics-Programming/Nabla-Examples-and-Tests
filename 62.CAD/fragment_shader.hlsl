#pragma shader_stage(fragment)

#include "common.hlsl"
#include <nbl/builtin/hlsl/shapes/rounded_line.hlsl>
#include <nbl/builtin/hlsl/shapes/ellipse.hlsl>

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

    ObjectType objType = (ObjectType)input.lineWidth_eccentricity_objType_writeToAlpha.z;

    float localAlpha = 0.0f;
    bool writeToAlpha = input.lineWidth_eccentricity_objType_writeToAlpha.w == 1u;
    
    if (writeToAlpha)
    {
        if (objType == ObjectType::ELLIPSE)
        {
            const float2 center = input.start_end.xy;
            const float2 majorAxis = input.start_end.zw;
            const float lineThickness = asfloat(input.lineWidth_eccentricity_objType_writeToAlpha.x) / 2.0f;
            const float eccentricity = (float)(input.lineWidth_eccentricity_objType_writeToAlpha.y) / UINT32_MAX;
            const float2 ellipseBounds = input.ellipseBounds_bezierP3P4.xy;
            float distance = nbl::hlsl::shapes::EllipseOutlineBounded_t::construct(center, majorAxis, ellipseBounds, eccentricity, lineThickness).signedDistance(input.position.xy);

            const float antiAliasingFactor = globals.antiAliasingFactor;
            localAlpha = 1.0f - smoothstep(-antiAliasingFactor, +antiAliasingFactor, distance);
        }
        else if (objType == ObjectType::LINE)
        {
            const float2 start = input.start_end.xy;
            const float2 end = input.start_end.zw;
            const float lineThickness = asfloat(input.lineWidth_eccentricity_objType_writeToAlpha.x) / 2.0f;
            
            float distance = nbl::hlsl::shapes::RoundedLine_t::construct(start, end, lineThickness).signedDistance(input.position.xy);

            /* No need to mul with fwidth(distance), distance already in screen space */
            const float antiAliasingFactor = globals.antiAliasingFactor;
            localAlpha = 1.0f - smoothstep(-antiAliasingFactor, +antiAliasingFactor, distance);
        }
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

    return float4(input.color.xyz, input.color.w * alpha);
}
