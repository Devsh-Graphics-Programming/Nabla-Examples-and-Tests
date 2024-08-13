// Copyright (C) 2024-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/builtin/hlsl/luma_meter/luma_meter.hlsl"
#include "nbl/builtin/hlsl/bda/bda_accessor.hlsl"
#include "app_resources/common.hlsl"

[[vk::combinedImageSampler]] [[vk::binding(0, 0)]] Texture2D texture;
[[vk::combinedImageSampler]] [[vk::binding(0, 0)]] SamplerState samplerState;

[[vk::push_constant]] AutoexposurePushData pushData;

using Ptr = nbl::hlsl::bda::__ptr < uint32_t >;
using PtrAccessor = nbl::hlsl::BdaAccessor < uint32_t >;

groupshared float32_t sdata[WorkgroupSize];

struct SharedAccessor
{
    uint32_t get(const uint32_t index)
    {
        return sdata[index];
    }

    void set(const uint32_t index, const uint32_t value)
    {
        sdata[index] = value;
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        nbl::hlsl::glsl::barrier();
    }
};

struct TexAccessor
{
    float32_t3 get(float32_t2 uv) {
        return texture.Sample(samplerState, uv).rgb;
    }
};

uint32_t3 nbl::hlsl::glsl::gl_WorkGroupSize()
{
    return uint32_t3(WorkgroupSize, 1, 1);
}

[numthreads(DeviceSubgroupSize, DeviceSubgroupSize, 1)]
void main(uint32_t3 ID : SV_GroupThreadID, uint32_t3 GroupID : SV_GroupID)
{
    nbl::hlsl::luma_meter::LumaMeteringWindow luma_meter_window;
    luma_meter_window.meteringWindowScale = float32_t2(pushData.meteringWindowScaleX, pushData.meteringWindowScaleY);
    luma_meter_window.meteringWindowOffset = float32_t2(pushData.meteringWindowOffsetX, pushData.meteringWindowOffsetY);

    const Ptr val_ptr = Ptr::create(pushData.lumaMeterBDA);
    PtrAccessor val_accessor = PtrAccessor::create(val_ptr);

    SharedAccessor sdata;
    TexAccessor tex;

    using LumaMeter = nbl::hlsl::luma_meter::geom_luma_meter< WorkgroupSize, PtrAccessor, SharedAccessor, TexAccessor>;
    LumaMeter meter = LumaMeter::create(luma_meter_window, pushData.lumaMin, pushData.lumaMax);

    uint32_t2 sampleCount = uint32_t2(pushData.sampleCountX, pushData.sampleCountY);
    uint32_t2 viewportSize = uint32_t2(pushData.viewportSizeX, pushData.viewportSizeY);

    meter.gatherLuma(val_accessor, tex, sdata, sampleCount, viewportSize);
}
