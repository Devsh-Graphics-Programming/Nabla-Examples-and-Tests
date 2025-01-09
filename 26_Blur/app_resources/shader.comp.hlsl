#include "common.hlsl"

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/limits.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/workgroup/scratch_size.hlsl"

using namespace nbl::hlsl;

enum EdgeWrapMode {
	WRAP_MODE_CLAMP_TO_EDGE,
	WRAP_MODE_CLAMP_TO_BORDER,
	WRAP_MODE_REPEAT,
	WRAP_MODE_MIRROR,
};

template<typename TextureAccessor, typename SharedAccessor>
void boxBlur(TextureAccessor texAccessor, SharedAccessor smemAccessor, EdgeWrapMode wrapMode, float32_t4 borderColor, float32_t radius, bool flip) {
    uint16_t2 pixelPos = uint16_t2(spirv::GlobalInvocationId.xy);
    if (flip) pixelPos = pixelPos.yx;
    float32_t scale = 1.0 / (2.0 * radius + 1.0);
    uint16_t radiusFl = (uint16_t)floor(radius);
    float32_t alpha = radius - radiusFl;

    float32_t4 color = 0;
    for (uint16_t ch = 0; ch < 4; ch++)
    {
        if (spirv::LocalInvocationId.x == 0)
        {
            for (uint16_t i = 0; i < glsl::gl_WorkGroupSize().x + (radiusFl + 1) * 2; i++)
            {
                uint16_t2 pos = pixelPos;
                pos[flip] = max((uint16_t)0, pos[flip] + i - radiusFl - 1);
                smemAccessor.set(i, texAccessor.get(pos, ch));
            }
        }

        glsl::barrier();

        uint16_t scanlineIdx = spirv::LocalInvocationId.x + radiusFl + 1;
        float32_t sum = smemAccessor.get(scanlineIdx);
        for (uint16_t j = 1; j <= radiusFl; j++)
        {
            sum += smemAccessor.get(scanlineIdx - j) + smemAccessor.get(scanlineIdx + j);
        }

        uint16_t last = texAccessor.size()[flip];
        float32_t left = spirv::WorkgroupId.x * glsl::gl_WorkGroupSize().x + float32_t(spirv::LocalInvocationId.x) - radiusFl - 1;
        float32_t right = spirv::WorkgroupId.x * glsl::gl_WorkGroupSize().x + float32_t(spirv::LocalInvocationId.x) + radiusFl;

        if (right < last)
        {
            sum += lerp(smemAccessor.get(scanlineIdx + radiusFl + 1), smemAccessor.get(scanlineIdx + radiusFl + 2), alpha);
        }
        else switch (wrapMode)
        {
            case WRAP_MODE_CLAMP_TO_EDGE:
                sum += (right - float32_t(last)) * (smemAccessor.get(glsl::gl_WorkGroupSize().x) - smemAccessor.get(glsl::gl_WorkGroupSize().x - 1));
            break;
            case WRAP_MODE_CLAMP_TO_BORDER:
                sum += (right - float32_t(last)) * borderColor[ch];
            break;
        }

        if (left > 0)
        {
            sum -= lerp(smemAccessor.get(scanlineIdx - radiusFl), smemAccessor.get(scanlineIdx - radiusFl - 1), alpha);
        }
        else switch (wrapMode)
        {
            case WRAP_MODE_CLAMP_TO_EDGE:
                sum -= (1 - abs(left)) * smemAccessor.get(0);
            break;
            case WRAP_MODE_CLAMP_TO_BORDER:
                sum -= (left + 1) * borderColor[ch];
            break;
        }

        color[ch] = sum;
    }

    texAccessor.set(pixelPos, color * scale);
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

    uint16_t2 size() {
        uint32_t3 dims;
        input.GetDimensions(0, dims.x, dims.y, dims.z);
        return uint32_t2(dims.xy);
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
void main() {
    TextureProxy texAccessor;
    SharedMemoryProxy smemAccessor;
    boxBlur(texAccessor, smemAccessor, WRAP_MODE_CLAMP_TO_EDGE, float32_t4(1, 0, 1, 1), 6, pc.flip);
}
