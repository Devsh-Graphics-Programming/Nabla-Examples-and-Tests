#pragma shader_stage(fragment)

#include "common.hlsl"

#if defined(NBL_FEATURE_FRAGMENT_SHADER_PIXEL_INTERLOCK)
[[vk::ext_instruction(/* OpBeginInvocationInterlockEXT */ 5364)]]
void beginInvocationInterlockEXT();
[[vk::ext_instruction(/* OpEndInvocationInterlockEXT */ 5365)]]
void endInvocationInterlockEXT();
#endif

float4 main(float4 position : SV_Position) : SV_TARGET
{
#if defined(NBL_FEATURE_FRAGMENT_SHADER_PIXEL_INTERLOCK)
    [[vk::ext_capability(/*FragmentShaderPixelInterlockEXT*/ 5378)]]
    [[vk::ext_extension("SPV_EXT_fragment_shader_interlock")]]
    vk::ext_execution_mode(/*PixelInterlockOrderedEXT*/ 5366);
#endif

    uint2 fragCoord = uint2(position.xy);
    
#if defined(NBL_FEATURE_FRAGMENT_SHADER_PIXEL_INTERLOCK)
    beginInvocationInterlockEXT();
    const uint packedData = pseudoStencil[fragCoord];
    pseudoStencil[fragCoord] = bitfieldInsert(0, InvalidMainObjectIdx, AlphaBits, MainObjectIdxBits);
    endInvocationInterlockEXT();
    
    const uint quantizedAlpha = bitfieldExtract(packedData,0,AlphaBits);
    const uint mainObjectIdx = bitfieldExtract(packedData,AlphaBits,MainObjectIdxBits);
    // draw with previous geometry's style :kek:
    float4 color = lineStyles[mainObjects[mainObjectIdx].styleIdx].color;
    color.a *= float(quantizedAlpha)/255.f;
    return color;
#else 
    return float4(0.0f, 0.0f, 0.0f, 0.0f);    
#endif
}