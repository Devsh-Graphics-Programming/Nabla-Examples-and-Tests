#include "common.hlsl"
#include <nbl/builtin/hlsl/spirv_intrinsics/fragment_shader_pixel_interlock.hlsl>

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

    nbl::hlsl::spirv::beginInvocationInterlockEXT();
    
    bool resolve = false;
    uint32_t toResolveStyleIdx = InvalidStyleIdx;
    const uint32_t packedData = pseudoStencil[fragCoord];
    const uint32_t storedQuantizedAlpha = nbl::hlsl::glsl::bitfieldExtract<uint32_t>(packedData,0,AlphaBits);
    const uint32_t storedMainObjectIdx = nbl::hlsl::glsl::bitfieldExtract<uint32_t>(packedData,AlphaBits,MainObjectIdxBits);

    const bool currentlyActiveMainObj = (storedMainObjectIdx == globals.currentlyActiveMainObjectIndex);
    if (!currentlyActiveMainObj)
    {
        // Normal Scenario, this branch will always be taken if there is no overflow submit in the middle of an active mainObject
        //we do the final resolve of the pixel and invalidate the pseudo-stencil
        pseudoStencil[fragCoord] = nbl::hlsl::glsl::bitfieldInsert<uint32_t>(0, InvalidMainObjectIdx, AlphaBits, MainObjectIdxBits);
        
        // if geomID has changed, we resolve the SDF alpha (draw using blend), else accumulate
        resolve = storedMainObjectIdx != InvalidMainObjectIdx;
    
        // load from colorStorage only if we want to resolve color from texture instead of style
        // sampling from colorStorage needs to happen in critical section because another fragment may also want to store into it at the same time + need to happen before store
        if (resolve)
        {
            toResolveStyleIdx = loadMainObject(storedMainObjectIdx).styleIdx;
            if (toResolveStyleIdx == InvalidStyleIdx) // if style idx to resolve is invalid, then it means we should resolve from color
                color = float32_t4(unpackR11G11B10_UNORM(colorStorage[fragCoord]), 1.0f);
        }
    }
    else if (globals.currentlyActiveMainObjectIndex != InvalidMainObjectIdx)
    {
        // Being here means there was an overflow submit in the middle of an active main objejct
        // We don't want to resolve the active mainObj, because it needs to fully resolved later when the mainObject  actually finishes.
        // We change the active main object index in our pseudo-stencil to 0u, because that will be it's new index in the next submit.
        uint32_t newMainObjectIdx = 0u;
        pseudoStencil[fragCoord] = nbl::hlsl::glsl::bitfieldInsert<uint32_t>(storedQuantizedAlpha, newMainObjectIdx, AlphaBits, MainObjectIdxBits);
        resolve = false; // just to re-iterate that we don't want to resolve this.
    }
    

    nbl::hlsl::spirv::endInvocationInterlockEXT();

    if (!resolve)
        discard;


    // draw with previous geometry's style's color or stored in texture buffer :kek:
    // we don't need to load the style's color in critical section because we've already retrieved the style index from the stored main obj
    if (toResolveStyleIdx != InvalidStyleIdx) // if toResolveStyleIdx is valid then that means our resolved color should come from line style
    {
        color = loadLineStyle(toResolveStyleIdx).color;
        gammaUncorrect(color.rgb); // want to output to SRGB without gamma correction
    }

    color.a *= float(storedQuantizedAlpha) / 255.f;
    
    return color;
}

[[vk::spvexecutionmode(spv::ExecutionModePixelInterlockOrderedEXT)]]
[shader("pixel")]
float4 resolveAlphaMain(float4 position : SV_Position) : SV_TARGET
{
    return calculateFinalColor<DeviceConfigCaps::fragmentShaderPixelInterlock>(position.xy);
}
