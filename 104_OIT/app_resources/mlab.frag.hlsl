#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include <nbl/builtin/hlsl/ext/FullScreenTriangle/SVertexAttributes.hlsl>
#include "nbl/builtin/hlsl/spirv_intrinsics/fragment_shader_pixel_interlock.hlsl"
#include "nbl/builtin/hlsl/limits.hlsl"
#include "app_resources/common.hlsl"

using namespace nbl::hlsl;
using namespace nbl::hlsl::ext::FullScreenTriangle;

[[vk::push_constant]] nbl::hlsl::this_example::SPushConstants pc;

[shader("pixel")]
float32_t4 PSMain(SVertexAttributes vxAttr, float32_t4 position : SV_Position) : SV_Target0
{
	spirv::execution_mode::PixelInterlockOrderedEXT();

	const uint32_t pixelPos = _static_cast<uint32_t>(position.y) * pc.width + _static_cast<uint32_t>(position.x);
	const uint64_t pixelBase = pc.bdaAddress + uint64_t(pixelPos) * (sizeof(Smlab) * 4);
    Smlab datas[4];
	uint16_t validData = 0;
	spirv::beginInvocationInterlockEXT();
    {
	for (uint32_t i = 0; i < 4; ++i)
    {
        datas[i] = vk::RawBufferLoad<Smlab>(pixelBase + i * sizeof(Smlab));
		if (datas[i].depth != numeric_limits<float>::max)
			validData++;
			
    }
	if (datas[1].depth == numeric_limits<float>::max)
	{
		return float32_t4(0,0,0,0);
	}
	for (uint32_t i = 0; i < validData - 1 ; ++i)
    {
		 const float32_t4 src = unpackUnorm4x8(int32_t(datas[i].colorDepth));
		 const float32_t4 dst = unpackUnorm4x8(int32_t(datas[i+1].colorDepth));

		 float32_t4 merged;
		 merged.rgb = src.rgb + dst.rgb * (1.f - src.a);
		 merged.a = src.a + dst.a * (1.f - src.a);

		 datas[i+1].colorDepth = _static_cast<uint32_t>(packUnorm4x8(merged));
	}

	return unpackUnorm4x8(int32_t(datas[validData - 1].colorDepth));
    spirv::endInvocationInterlockEXT();
	}
}