#include "common.hlsl"

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "nbl/builtin/hlsl/workgroup/arithmetic.hlsl"
#include "nbl/builtin/hlsl/workgroup/scratch_size.hlsl"

using namespace nbl::hlsl;

template<class TextureAccessor, class SharedAccessor>
void boxBlur(
    NBL_REF_ARG(TextureAccessor) texAccessor,
    NBL_REF_ARG(SharedAccessor) smemAccessor,
    uint16_t itemsPerThread,
    uint16_t width,
    uint16_t channel,
    uint32_t wrapMode,
    float32_t4 borderColor,
    float32_t radius,
    bool flip
) {
    const uint16_t lastIdx = width - 1;
    const float32_t scale = 1.0 / (2.0 * radius + 1.0);
    const uint16_t radiusFl = (uint16_t)floor(radius);
    const uint16_t radiusCl = (uint16_t)ceil(radius);
    const float32_t alpha = radius - radiusFl;
    
    for (uint16_t i = 0; i < itemsPerThread; i++)
    {
        uint16_t scanIdx = (i * glsl::gl_WorkGroupSize().x) + workgroup::SubgroupContiguousIndex();
        if (scanIdx > lastIdx) break;

        float32_t sum = texAccessor.get(i, channel);
        if (i != 0) {
            glsl::barrier();
            if (workgroup::SubgroupContiguousIndex() == 0) {
                float32_t previusSum = 0;
                smemAccessor.get(scanIdx - 1, previusSum);
                sum += previusSum;
            }
        }
        sum = workgroup::inclusive_scan<plus<float32_t>, WORKGROUP_SIZE>::template __call<SharedAccessor>(sum, smemAccessor);
        smemAccessor.set(scanIdx, sum);
        glsl::barrier();
    }

    for (uint16_t i = 0; i < itemsPerThread; i++)
    {
        uint16_t scanIdx = (i * glsl::gl_WorkGroupSize().x) + workgroup::SubgroupContiguousIndex();
        if (scanIdx > lastIdx) break;
        
        const float32_t leftIdx = scanIdx - radius;
        const float32_t rightIdx = scanIdx + radius;

        float32_t result = 0;
        if (rightIdx <= lastIdx)
        {
            float32_t rightFloor, rightCeil;
            smemAccessor.get(scanIdx + radiusFl, rightFloor);
            smemAccessor.get(scanIdx + radiusCl, rightCeil);
            result += lerp(rightFloor, rightCeil, alpha);
        }
        else switch (wrapMode)
        {
            case WRAP_MODE_CLAMP_TO_BORDER:
            {
                result += (rightIdx - lastIdx) * borderColor[channel];
            } break;
            case WRAP_MODE_CLAMP_TO_EDGE:
            {
                float32_t last, lastMinus1;
                smemAccessor.get(lastIdx, last);
                smemAccessor.get(lastIdx - 1, lastMinus1);
                result += (rightIdx - lastIdx) * (last - lastMinus1);
            } break;
            case WRAP_MODE_REPEAT:
            {
                const uint16_t repeatedIdx = (uint16_t)(rightIdx % width);
                float32_t repeatedValue;
                smemAccessor.get(repeatedIdx, repeatedValue);
                result += repeatedValue;
            } break;
            case WRAP_MODE_MIRROR:
            {        
                const uint16_t mirroredIdx = (uint16_t)((rightIdx / width) % 2 == 0 ? rightIdx % width : lastIdx - (rightIdx % width));
                float32_t mirrored;
                smemAccessor.get(mirroredIdx, mirrored);
                result += mirrored;
            } break;
        }

        if (leftIdx >= 0)
        {
            float32_t leftFloor, leftCeil;
            smemAccessor.get(scanIdx - radiusFl, leftFloor);
            smemAccessor.get(scanIdx - radiusCl, leftCeil);
            result -= lerp(leftFloor, leftCeil, alpha);
        }
        else switch (wrapMode)
        {
            case WRAP_MODE_CLAMP_TO_BORDER:
            {
                result -= leftIdx * borderColor[channel];
            } break;
            case WRAP_MODE_CLAMP_TO_EDGE:
            {
                float32_t first;
                smemAccessor.get(0, first);
                result -= abs(leftIdx) * first;
            } break;
            case WRAP_MODE_REPEAT:
            {
                const uint16_t repeatedIdx = (uint16_t)((leftIdx % width + width) % width);
                float32_t repeatedValue;
                smemAccessor.get(repeatedIdx, repeatedValue);
                result -= repeatedValue;
            } break;
            case WRAP_MODE_MIRROR:
            {
                const uint16_t mirroredIdx = (uint16_t)((-leftIdx / width) % 2 == 0 ? (-leftIdx) % width : lastIdx - ((-leftIdx) % width));
                float32_t mirrored;
                smemAccessor.get(mirroredIdx, mirrored);
                result -= mirrored;
            } break;
        }
        
        texAccessor.set(i, channel, result * scale);
    }
}

static const uint32_t scratchSize = workgroup::scratch_size_arithmetic<WORKGROUP_SIZE>::value;
static const uint32_t smemSize = WORKGROUP_SIZE + scratchSize; // TODO: try
uint32_t3 glsl::gl_WorkGroupSize() { return uint32_t3(WORKGROUP_SIZE, 1, 1); }

[[vk::binding(0)]]
Texture2D<float32_t4> input;
[[vk::binding(1)]]
RWTexture2D<float32_t4> output;

[[vk::push_constant]] PushConstants pc;

groupshared float32_t smem[4096];

template<uint16_t SpillSize>
struct TextureProxy
{
	void load(uint16_t i)
	{
		spill[i] = input[position(i)];
	}

	void store(uint16_t i)
	{
		output[position(i)] = spill[i];
	}

	float32_t get(uint16_t i, uint16_t ch)
	{
		return spill[i][ch];
	}

	void set(uint16_t i, uint16_t ch, float32_t value)
	{
        spill[i][ch] = value;
	}

    uint16_t2 position(uint16_t i) {
        uint32_t2 pos;
        pos.x = (i * glsl::gl_WorkGroupSize().x) + workgroup::SubgroupContiguousIndex();
        pos.y = spirv::WorkgroupId.x;
        return pc.flip ? pos.yx : pos.xy;
    }

    uint16_t2 size() {
        uint32_t3 dims;
        input.GetDimensions(0, dims.x, dims.y, dims.z);
        return uint32_t2(dims.xy);
    }

    float32_t4 spill[SpillSize];
};

struct SharedMemoryProxy
{
	void get(const uint16_t idx, NBL_REF_ARG(float32_t) value)
	{
		value = smem[idx];
	}

	void set(const uint16_t idx, float32_t value)
	{
        smem[idx] = value;
	}

    void workgroupExecutionAndMemoryBarrier()
    {
        glsl::barrier();
    }
};

[numthreads(WORKGROUP_SIZE, 1, 1)]
void main() {
    TextureProxy<32> texAccessor;
    SharedMemoryProxy smemAccessor;
    uint16_t width = texAccessor.size()[pc.flip];
    uint16_t itemsPerThread = (uint16_t)ceil((float32_t)width / (float32_t)WORKGROUP_SIZE);
    
    for (uint16_t i = 0; i < itemsPerThread; i++)
        texAccessor.load(i);
    glsl::barrier();

    for (uint16_t ch = 0; ch < 4; ch++)
    {
        boxBlur(texAccessor, smemAccessor, itemsPerThread, width, ch, pc.edgeWrapMode, float32_t4(0, 1, 0, 1), pc.radius, pc.flip);
    }

    glsl::barrier();
    for (uint16_t i = 0; i < itemsPerThread; i++)
        texAccessor.store(i);
}
