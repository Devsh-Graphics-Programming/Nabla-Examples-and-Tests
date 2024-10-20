#pragma shader_stage(fragment)

#include "common.hlsl"
#include <nbl/builtin/hlsl/spirv_intrinsics/fragment_shader_pixel_interlock.hlsl>
#include <nbl/builtin/hlsl/jit/device_capabilities.hlsl>

template<bool FragmentShaderPixelInterlock>
float32_t4 calculateFinalColor(const uint2 fragCoord);


template<>
float32_t4 calculateFinalColor<false>(const uint2 fragCoord)
{
    return float4(0.0f, 0.0f, 0.0f, 0.0f);
}

template<>
float32_t4 calculateFinalColor<true>(const uint2 fragCoord)
{
    float32_t4 color;
    
    nbl::hlsl::spirv::execution_mode::PixelInterlockOrderedEXT();
    nbl::hlsl::spirv::beginInvocationInterlockEXT();

    const uint32_t packedData = pseudoStencil[fragCoord];
    const uint32_t storedQuantizedAlpha = nbl::hlsl::glsl::bitfieldExtract<uint32_t>(packedData,0,AlphaBits);
    const uint32_t storedMainObjectIdx = nbl::hlsl::glsl::bitfieldExtract<uint32_t>(packedData,AlphaBits,MainObjectIdxBits);
    pseudoStencil[fragCoord] = nbl::hlsl::glsl::bitfieldInsert<uint32_t>(0, InvalidMainObjectIdx, AlphaBits, MainObjectIdxBits);
    
    uint32_t resolveStyleIdx = mainObjects[storedMainObjectIdx].styleIdx;
    const bool resolveColorFromStyle = resolveStyleIdx != InvalidStyleIdx;
    if (!resolveColorFromStyle)
        color = float32_t4(unpackR11G11B10_UNORM(colorStorage[fragCoord]), 1.0f);

    nbl::hlsl::spirv::endInvocationInterlockEXT();

    if (resolveColorFromStyle)
        color = lineStyles[resolveStyleIdx].color;
    color.a *= float(storedQuantizedAlpha) / 255.f;
    
    return color;
}

float4 main(float4 position : SV_Position) : SV_TARGET
{
    return calculateFinalColor<nbl::hlsl::jit::device_capabilities::fragmentShaderPixelInterlock>(position.xy);
}
