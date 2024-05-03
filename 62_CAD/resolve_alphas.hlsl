#pragma shader_stage(fragment)

#include "common.hlsl"
#include <nbl/builtin/hlsl/spirv_intrinsics/fragment_shader_pixel_interlock.hlsl>
#include <nbl/builtin/hlsl/jit/device_capabilities.hlsl>

template<bool FragmentShaderPixelInterlock>
float32_t4 calculateColor(const uint2 fragCoord);

template<>
float32_t4 calculateColor<false>(const uint2 fragCoord)
{
    return float4(0.0f, 0.0f, 0.0f, 0.0f);
}

template<>
float32_t4 calculateColor<true>(const uint2 fragCoord)
{
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
}

float4 main(float4 position : SV_Position) : SV_TARGET
{
    return calculateColor<nbl::hlsl::jit::device_capabilities::fragmentShaderPixelInterlock>(position.xy);
}
