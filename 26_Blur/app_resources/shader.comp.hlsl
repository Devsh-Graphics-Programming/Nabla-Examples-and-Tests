#include "common.hlsl"

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/limits.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/workgroup/scratch_size.hlsl"
// #include "nbl/builtin/hlsl/workgroup/arithmetic.hlsl"
// #include "nbl/builtin/hlsl/workgroup/shared_scan.hlsl"
// #include "nbl/builtin/hlsl/colorspace/OETF.hlsl"

using namespace nbl::hlsl;

template<typename TextureAccessor, typename SharedAccessor>
void boxBlur(TextureAccessor texAccessor, SharedAccessor smemAccessor, float32_t radius, bool flip) {
    uint16_t2 pixel_pos = uint16_t2(spirv::GlobalInvocationId.xy);
    if (flip) pixel_pos = pixel_pos.yx;
    float32_t scale = 1.0 / (2.0 * radius + 1.0);
    uint16_t radius_f = (uint16_t)floor(radius);
    float32_t alpha = radius - radius_f;

    float32_t4 color = 0;
    for (uint16_t ch = 0; ch < 4; ch++)
    {
        if (spirv::LocalInvocationId.x == 0)
        {
            for (uint16_t i = 0; i < glsl::gl_WorkGroupSize().x + (radius_f + 1) * 2; i++)
            {
                uint16_t2 pos = pixel_pos;
                pos[flip] = max((uint16_t)0, pos[flip] + i - radius_f - 1);
                smemAccessor.set(i, texAccessor.get(pos, ch));
            }
        }

        glsl::barrier();

        uint16_t scanline_idx = radius_f + 1 + spirv::LocalInvocationId.x;
        float32_t sum = smemAccessor.get(scanline_idx);
        for (uint16_t j = 1; j <= radius_f; j++)
        {
            sum += smemAccessor.get(scanline_idx - j) + smemAccessor.get(scanline_idx + j);
        }
        sum += alpha * (smemAccessor.get(scanline_idx - radius_f - 1) + smemAccessor.get(scanline_idx + radius_f + 1));

        sum += lerp(smemAccessor.get(scanline_idx + radius_f + 1), smemAccessor.get(scanline_idx + radius_f + 2), alpha);
        sum -= lerp(smemAccessor.get(scanline_idx - radius_f), smemAccessor.get(scanline_idx - radius_f - 1), alpha);

        color[ch] = sum;
    }

    texAccessor.set(pixel_pos, color * scale);
}

static const uint32_t arithmeticSz = workgroup::scratch_size_arithmetic<WORKGROUP_SIZE>::value;
static const uint32_t smemSize = WORKGROUP_SIZE + arithmeticSz;
uint32_t3 glsl::gl_WorkGroupSize() { return uint32_t3( WORKGROUP_SIZE, 1, 1 ); }

[[vk::binding(0)]]
Texture2D<float32_t4> input;
[[vk::binding(1)]]
RWTexture2D<float32_t4> output;

[[vk::push_constant]] PushConstants pc;

groupshared float32_t smem[smemSize];

struct TextureProxy
{
	float32_t get(const uint16_t2 pos, const uint16_t ch)
	{
		return input[pos][ch];
	}

	void set(const uint16_t2 pos, float32_t4 value)
	{
        output[pos] = value;
	}
};

struct SharedMemoryProxy
{
	float32_t get(const uint16_t idx)
	{
		return smem[idx];
	}

	void set(const uint16_t idx, float32_t value)
	{
        smem[idx] = value;
	}
};

[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint3 group_thread_id: SV_GroupThreadID, uint3 dispatch_thread_id: SV_DispatchThreadID) {
    TextureProxy texAccessor;
    SharedMemoryProxy smemAccessor;
    boxBlur(texAccessor, smemAccessor, 6, pc.flip);
}
