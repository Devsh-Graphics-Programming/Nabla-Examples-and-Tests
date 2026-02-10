// Copyright (C) 2024-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/builtin/hlsl/luma_meter/geom_mean.hlsl"
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
        return textureIn.SampleLevel(samplerStateIn, uv, 0.f).rgb;
    }
};

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

    float32_t EV = meter.gatherLuma(val_accessor);

    const float32_t lumaDiff = vk::RawBufferLoad<float32_t>(pushData.pLastFrameEVBuf) - EV;
    EV += lumaDiff * mix(pushData.exposureAdaptationFactors.x, pushData.exposureAdaptationFactors.y, lumaDiff >= 0.0);

    uint32_t tid = workgroup::SubgroupContiguousIndex();
    if (all(glsl::gl_WorkGroupID() == uint32_t3(0,0,0)))
        if (tid == 0)
            vk::RawBufferStore<float32_t>(pushData.pCurrFrameEVBuf, EV);

    morton::code<false, 32, 2> mc;
    mc.value = tid;
    uint32_t2 coord = _static_cast<uint32_t2>(mc);

    uint32_t2 pos = (glsl::gl_WorkGroupID() * SUBGROUP_SIZE).xy + coord;
    if (any(pos < promote<uint32_t2>(0u)) || any(pos >= pushData.viewportSize))
        return;

    float32_t2 uv = float32_t2(pos) / pushData.viewportSize;
    float32_t3 color = colorspace::eotf::sRGB(tex.get(uv).rgb);
    float32_t3 CIEColor = mul(colorspace::sRGBtoXYZ, color);
    tonemapper::Reinhard<float32_t> reinhard = tonemapper::Reinhard<float32_t>::create(EV, 1.0f, 0.85f);
    const float32_t ditherFactor = 0.5f;    // TODO: dithering
    float32_t3 tonemappedColor = mul(colorspace::decode::XYZtoscRGB, reinhard(CIEColor)*ditherFactor);

    textureOut[pos] = float32_t4(tonemappedColor, 1.0f);
}
