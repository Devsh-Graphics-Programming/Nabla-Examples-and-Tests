// Copyright (C) 2024-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/builtin/hlsl/luma_meter/luma_meter.hlsl"
#include "nbl/builtin/hlsl/bda/bda_accessor.hlsl"
#include "nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl"
#include "app_resources/common.hlsl"

[[vk::combinedImageSampler]] [[vk::binding(0, 0)]] Texture2D texture;
[[vk::combinedImageSampler]] [[vk::binding(0, 0)]] SamplerState samplerState;

using namespace nbl::hlsl;
using Ptr = bda::__ptr < uint32_t >;
using PtrAccessor = BdaAccessor < uint32_t >;

[[vk::push_constant]] AutoexposurePushData pushData;

groupshared float32_t sdata[WorkgroupSize];
struct SharedAccessor
{
    using type = float32_t;
    void get(const uint32_t index, NBL_REF_ARG(uint32_t) value)
    {
        value = sdata[index];
    }

    void set(const uint32_t index, const uint32_t value)
    {
        sdata[index] = value;
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        glsl::barrier();
    }
};

struct TexAccessor
{
    static float32_t3 toXYZ(float32_t3 srgbColor) {
        return mul(colorspace::sRGBtoXYZ, srgbColor);
    }

    float32_t3 get(float32_t2 uv) {
        return texture.SampleLevel(samplerState, uv, 0.f).rgb;
    }
};

uint32_t3 glsl::gl_WorkGroupSize()
{
    return uint32_t3(DeviceSubgroupSize, DeviceSubgroupSize, 1);
}

[numthreads(DeviceSubgroupSize, DeviceSubgroupSize, 1)]
[shader("compute")]
void main(uint32_t3 ID : SV_GroupThreadID, uint32_t3 GroupID : SV_GroupID)
{
    const Ptr val_ptr = Ptr::create(pushData.lumaMeterBDA);
    PtrAccessor val_accessor = PtrAccessor::create(val_ptr);

    SharedAccessor sdata;
    TexAccessor tex;

    using LumaMeter = luma_meter::geom_meter< WorkgroupSize, PtrAccessor, SharedAccessor, TexAccessor>;
    LumaMeter meter = LumaMeter::create(pushData.lumaMinMax, pushData.sampleCount);

    meter.sampleLuma(pushData.window, val_accessor, tex, sdata, (float32_t2)(glsl::gl_WorkGroupID() * glsl::gl_WorkGroupSize()), pushData.viewportSize);
}
