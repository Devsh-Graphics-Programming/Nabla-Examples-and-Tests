// Copyright (C) 2024-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/builtin/hlsl/luma_meter/geom_mean.hlsl"
#include "nbl/builtin/hlsl/bda/bda_accessor.hlsl"
#include "nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl"
#include "app_resources/common.hlsl"

[[vk::combinedImageSampler]] [[vk::binding(0, 0)]] Texture2D texture;
[[vk::combinedImageSampler]] [[vk::binding(0, 0)]] SamplerState samplerState;

using namespace nbl::hlsl;
using Ptr = bda::__ptr < uint32_t >;
using PtrAccessor = BdaAccessor < uint32_t >;

[[vk::push_constant]] AutoexposurePushData pushData;

groupshared float32_t sdata[WORKGROUP_SIZE];
struct SharedAccessor
{
    using type = float32_t;
    template<typename AccessType, typename IndexType>
    void get(const uint32_t ix, NBL_REF_ARG(AccessType) value)
    {
        value = sdata[ix];
    }
    template<typename AccessType, typename IndexType>
    void set(const uint32_t ix, const AccessType value)
    {
        sdata[ix] = value;
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        glsl::barrier();
    }
};

struct TexAccessor
{
    static float32_t3 toXYZ(float32_t3 srgbColor) {
        return dot(colorspace::sRGBtoXYZ[1], srgbColor);
    }

    float32_t3 get(float32_t2 uv) {
        return texture.SampleLevel(samplerState, uv, 0.f).rgb;
    }
};

uint32_t3 glsl::gl_WorkGroupSize()
{
    return uint32_t3(SUBGROUP_SIZE, SUBGROUP_SIZE, 1);
}

[numthreads(SUBGROUP_SIZE, SUBGROUP_SIZE, 1)]
[shader("compute")]
void main(uint32_t3 ID : SV_GroupThreadID, uint32_t3 GroupID : SV_GroupID)
{
    const Ptr val_ptr = Ptr::create(pushData.pLumaMeterBuf);
    PtrAccessor val_accessor = PtrAccessor::create(val_ptr);

    SharedAccessor sdata;
    TexAccessor tex;

    using LumaMeter = luma_meter::geom_meter<wg_config_t, PtrAccessor, SharedAccessor, TexAccessor, device_capabilities>;
    LumaMeter meter = LumaMeter::create(pushData.lumaMin, pushData.lumaMax, pushData.sampleCount, pushData.rcpFirstPassWGCount);

    uint32_t texWidth, texHeight;
    texture.GetDimensions(texWidth, texHeight);
    meter.sampleLuma(pushData.window, val_accessor, tex, sdata, (float32_t2)(glsl::gl_WorkGroupID() * glsl::gl_WorkGroupSize()), float32_t2(texWidth, texHeight));
}
