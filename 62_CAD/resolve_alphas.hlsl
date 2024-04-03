#pragma shader_stage(fragment)

#include "common.hlsl"
#include <nbl/builtin/hlsl/spirv_intrinsics/fragment_shader_pixel_interlock.hlsl>

float4 main(float4 position : SV_Position) : SV_TARGET
{
    uint2 fragCoord = uint2(position.xy);
    
#if 1
    nbl::hlsl::spirv::execution_mode::PixelInterlockOrderedEXT();
    
    nbl::hlsl::spirv::beginInvocationInterlockEXT();
    const uint packedData = pseudoStencil[fragCoord];
    pseudoStencil[fragCoord] = bitfieldInsert(0, InvalidMainObjectIdx, AlphaBits, MainObjectIdxBits);
    nbl::hlsl::spirv::endInvocationInterlockEXT();
    
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