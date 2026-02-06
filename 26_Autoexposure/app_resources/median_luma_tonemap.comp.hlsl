// Copyright (C) 2024-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/builtin/hlsl/luma_meter/histogram.hlsl"
#include "nbl/builtin/hlsl/bda/bda_accessor.hlsl"
#include "nbl/builtin/hlsl/colorspace/EOTF.hlsl"
#include "nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl"
#include "nbl/builtin/hlsl/colorspace/decodeCIEXYZ.hlsl"
#include "nbl/builtin/hlsl/colorspace/OETF.hlsl"
#include "nbl/builtin/hlsl/tonemapper/operators/reinhard.hlsl"
#include "app_resources/common.hlsl"

[[vk::combinedImageSampler]] [[vk::binding(0, 0)]] Texture2D textureIn;
[[vk::combinedImageSampler]] [[vk::binding(0, 0)]] SamplerState samplerStateIn;
[[vk::binding(0, 3)]] RWTexture2D<float32_t4> textureOut;

using namespace nbl::hlsl;
using Ptr = bda::__ptr < uint32_t >;
using PtrAccessor = BdaAccessor < uint32_t >;

[[vk::push_constant]] AutoexposurePushData pushData;

groupshared uint32_t sdata[BIN_COUNT];
struct SharedAccessor
{
    using type = uint32_t;
    template<typename AccessType, typename IndexType=uint32_t>
    void get(const uint32_t ix, NBL_REF_ARG(AccessType) value)
    {
        value = sdata[ix];
    }

    template<typename AccessType, typename IndexType=uint32_t>
    void set(const uint32_t ix, const AccessType value)
    {
        sdata[ix] = value;
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        glsl::barrier();
    }

    uint32_t atomicAdd(const uint32_t index, const uint32_t value)
    {
        return glsl::atomicAdd(sdata[index], value);
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
    return uint32_t3(SUBGROUP_SIZE, SUBGROUP_SIZE, 1);
}

[numthreads(SUBGROUP_SIZE, SUBGROUP_SIZE, 1)]
[shader("compute")]
void main(uint32_t3 ID : SV_GroupThreadID, uint32_t3 GroupID : SV_GroupID)
{
    const Ptr histo_ptr = Ptr::create(pushData.pLumaMeterBuf);
    PtrAccessor histo_accessor = PtrAccessor::create(histo_ptr);

    SharedAccessor sdata;
    TexAccessor tex;

    using LumaMeter = luma_meter::median_meter<wg_config_t, BIN_COUNT, PtrAccessor, SharedAccessor, TexAccessor, device_capabilities>;
    LumaMeter meter = LumaMeter::create(pushData.lumaMin, pushData.lumaMax, pushData.lowerBoundPercentile, pushData.upperBoundPercentile);

    float32_t EV = meter.gatherLuma(histo_accessor, sdata);

    const float32_t lumaDiff = vk::RawBufferLoad<float32_t>(pushData.pLastFrameEVBuf) - EV;
    EV += lumaDiff * mix(pushData.exposureAdaptationFactors.x, pushData.exposureAdaptationFactors.y, lumaDiff >= 0.0);

    uint32_t tid = workgroup::SubgroupContiguousIndex();
    if (all(glsl::gl_WorkGroupID() == uint32_t3(0,0,0)))
        if (tid == 0)
            vk::RawBufferStore<float32_t>(pushData.pLastFrameEVBuf, EV);

    morton::code<false, 32, 2> mc;
    mc.value = tid;
    uint32_t2 coord = _static_cast<uint32_t2>(mc);

    uint32_t2 pos = (glsl::gl_WorkGroupID() * glsl::gl_WorkGroupSize()).xy + coord;
    float32_t2 uv = (float32_t2)(pos) / pushData.viewportSize;
    float32_t3 color = colorspace::eotf::sRGB(tex.get(uv).rgb);
    float32_t3 CIEColor = mul(colorspace::sRGBtoXYZ, color);
    tonemapper::Reinhard<float32_t> reinhard = tonemapper::Reinhard<float32_t>::create(EV, 0.18, 0.85f);
    float32_t3 tonemappedColor = mul(colorspace::decode::XYZtoscRGB, reinhard(CIEColor));

    textureOut[pos] = float32_t4(colorspace::oetf::sRGB(tonemappedColor), 1.0f);
}
