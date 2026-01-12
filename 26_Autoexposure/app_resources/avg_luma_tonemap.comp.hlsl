// Copyright (C) 2024-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/builtin/hlsl/luma_meter/luma_meter.hlsl"
#include "nbl/builtin/hlsl/bda/bda_accessor.hlsl"
#include "nbl/builtin/hlsl/colorspace/EOTF.hlsl"
#include "nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl"
#include "nbl/builtin/hlsl/colorspace/decodeCIEXYZ.hlsl"
#include "nbl/builtin/hlsl/colorspace/OETF.hlsl"
#include "nbl/builtin/hlsl/tonemapper/operators.hlsl"
#include "app_resources/common.hlsl"

[[vk::combinedImageSampler]] [[vk::binding(0, 0)]] Texture2D textureIn;
[[vk::combinedImageSampler]] [[vk::binding(0, 0)]] SamplerState samplerStateIn;
[[vk::binding(0, 3)]] RWTexture2D<float32_t4> textureOut;

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
        return dot(colorspace::sRGBtoXYZ[1], srgbColor);
    }

    float32_t3 get(float32_t2 uv) {
        return textureIn.SampleLevel(samplerStateIn, uv, 0.f).rgb;
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

    float32_t EV = meter.gatherLuma(val_accessor);

    uint32_t tid = workgroup::SubgroupContiguousIndex();
    uint32_t2 coord = math::Morton<uint32_t>::decode2d(tid);

    uint32_t2 pos = (glsl::gl_WorkGroupID() * glsl::gl_WorkGroupSize()).xy + coord;
    float32_t2 uv = (float32_t2)(pos) / pushData.viewportSize;
    float32_t3 color = colorspace::oetf::sRGB(tex.get(uv).rgb);
    float32_t3 CIEColor = mul(colorspace::sRGBtoXYZ, color);
    tonemapper::Reinhard<float32_t> reinhard = tonemapper::Reinhard<float32_t>::create(EV, 0.18f, 0.85f);
    float32_t3 tonemappedColor = mul(colorspace::decode::XYZtoscRGB, reinhard(CIEColor));

    textureOut[pos] = float32_t4(tonemappedColor, 1.0f);
}
