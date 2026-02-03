// Copyright (C) 2024-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/builtin/hlsl/luma_meter/histogram.hlsl"
#include "nbl/builtin/hlsl/bda/bda_accessor.hlsl"
#include "nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl"
#include "app_resources/common.hlsl"

[[vk::combinedImageSampler]] [[vk::binding(0, 0)]] Texture2D texture;
[[vk::combinedImageSampler]] [[vk::binding(0, 0)]] SamplerState samplerState;

using namespace nbl::hlsl;
using Ptr = bda::__ptr < uint32_t >;
using PtrAccessor = BdaAccessor < uint32_t >;

[[vk::push_constant]] AutoexposurePushData pushData;

#define BIN_COUNT 1024

groupshared uint32_t sdata[BIN_COUNT];
struct SharedAccessor
{
    using type = uint32_t;
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

    float32_t atomicAdd(const uint32_t index, const uint32_t value) {
        return glsl::atomicAdd(sdata[index], value);
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
    const Ptr histo_ptr = Ptr::create(pushData.lumaMeterBDA);
    PtrAccessor histo_accessor = PtrAccessor::create(histo_ptr);

    SharedAccessor sdata;
    TexAccessor tex;

    using LumaMeter = luma_meter::median_meter< WORKGROUP_SIZE, BIN_COUNT, PtrAccessor, SharedAccessor, TexAccessor>;
    LumaMeter meter = LumaMeter::create(pushData.lumaMin, pushData.lumaMax, pushData.lowerBoundPercentile, pushData.upperBoundPercentile);

    uint32_t texWidth, texHeight;
    texture.GetDimensions(texWidth, texHeight);
    meter.sampleLuma(pushData.window, histo_accessor, tex, sdata, (float32_t2)(glsl::gl_WorkGroupID() * glsl::gl_WorkGroupSize()), float32_t2(texWidth, texHeight));
}
