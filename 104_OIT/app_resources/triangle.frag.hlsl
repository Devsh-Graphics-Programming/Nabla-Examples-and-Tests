// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/spirv_intrinsics/fragment_shader_pixel_interlock.hlsl"
#include "nbl/builtin/hlsl/limits.hlsl"
#include "app_resources/common.hlsl"

using namespace nbl::hlsl;
[[vk::push_constant]] nbl::hlsl::this_example::SPushConstants pc;
    
struct PSInput
{
    float32_t4 position : SV_Position;
    float32_t4 color : COLOR0;
};

[shader("pixel")]
void PSMain(PSInput input) : SV_Target
{
    spirv::execution_mode::PixelInterlockOrderedEXT();
    const uint32_t pixelPos = _static_cast<uint32_t>(input.position.y) * pc.width + _static_cast<uint32_t>(input.position.x);
    const float32_t pixelDepth = _static_cast<float32_t>(input.position.z);    
    const uint64_t pixelBase = pc.bdaAddress + uint64_t(pixelPos) * (sizeof(Smlab) * 4);
    const uint32_t packedColor = _static_cast<uint32_t>(packUnorm4x8(float32_t4(input.color)));
    Smlab inHandPixel = {pixelDepth,packedColor};
    spirv::beginInvocationInterlockEXT();
    {
        Smlab datas[4];
        for (uint32_t i = 0; i < 4; ++i)
        {
            datas[i] = vk::RawBufferLoad<Smlab>(pixelBase + i * sizeof(Smlab));
        }
        for (uint32_t i = 0; i < 4; ++i)
        {
            if (datas[i].depth == numeric_limits<float>::max)
            {
                vk::RawBufferStore<Smlab>(pixelBase + i * sizeof(Smlab),inHandPixel);
                inHandPixel.depth = numeric_limits<float>::max;
                break;
            }
            if (i == 3 && inHandPixel.depth != numeric_limits<float>::max)
            {
                const float32_t4 src = unpackUnorm4x8(int32_t(datas[i].colorDepth));
                const float32_t4 dst = unpackUnorm4x8(int32_t(inHandPixel.colorDepth));

                float32_t4 merged;
                merged.rgb = src.rgb + dst.rgb * (1.f - src.a);
                merged.a = src.a + dst.a * (1.f - src.a);

                Smlab store;
                store.depth = datas[i].depth;
                store.colorDepth = _static_cast<uint32_t>(packUnorm4x8(merged));
                vk::RawBufferStore<Smlab>(pixelBase + 3 * sizeof(Smlab), store);
                break;
            }
            if (inHandPixel.depth <= datas[i].depth)
            {
                vk::RawBufferStore<Smlab>(pixelBase + i * sizeof(Smlab),inHandPixel);
                inHandPixel = datas[i];
            }
        }
        
    }
    spirv::endInvocationInterlockEXT();
}
