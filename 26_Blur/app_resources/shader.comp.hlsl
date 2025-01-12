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
    uint32_t wrapMode,
    float32_t4 borderColor,
    uint16_t channels,
    float32_t radius,
    bool flip
) {
    const uint16_t n = texAccessor.size()[flip];
    const uint16_t lastIdx = n - 1;
    const uint16_t itemsPerThread = (uint16_t)ceil((float32_t)n / (float32_t)WORKGROUP_SIZE);

    const float32_t scale = 1.0 / (2.0 * radius + 1.0);
    const uint16_t radiusFl = (uint16_t)floor(radius);
    const uint16_t radiusCl = (uint16_t)ceil(radius);
    const float32_t alpha = radius - radiusFl;
    
    uint16_t2 pixelPos = 0;
    pixelPos[(flip + 1) % 2] = spirv::WorkgroupId.x;

    for (uint16_t i = 0; i < itemsPerThread; i++)
    {
        uint16_t scanIdx = (i * glsl::gl_WorkGroupSize().x) + workgroup::SubgroupContiguousIndex();
        if (scanIdx > lastIdx) break;
        pixelPos[flip] = scanIdx;
        
        const float32_t leftIdx = scanIdx - radius;
        const float32_t rightIdx = scanIdx + radius;
        float32_t4 color = 0;

        for (uint16_t ch = 0; ch < channels; ch++)
        {
            float32_t sum = texAccessor.get(pixelPos, ch);
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
                    result += (rightIdx - lastIdx) * borderColor[ch];
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
                    const uint16_t repeatedIdx = (uint16_t)(rightIdx % n);
                    float32_t repeatedValue;
                    smemAccessor.get(repeatedIdx, repeatedValue);
                    result += repeatedValue;
                } break;
                case WRAP_MODE_MIRROR:
                {        
                    const uint16_t mirroredIdx = (uint16_t)((rightIdx / n) % 2 == 0 ? rightIdx % n : lastIdx - (rightIdx % n));
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
                    result -= leftIdx * borderColor[ch];
                } break;
                case WRAP_MODE_CLAMP_TO_EDGE:
                {
                    float32_t first;
                    smemAccessor.get(0, first);
                    result -= abs(leftIdx) * first;
                } break;
                case WRAP_MODE_REPEAT:
                {
                    const uint16_t repeatedIdx = (uint16_t)((leftIdx % n + n) % n);
                    float32_t repeatedValue;
                    smemAccessor.get(repeatedIdx, repeatedValue);
                    result -= repeatedValue;
                } break;
                case WRAP_MODE_MIRROR:
                {
                    const uint16_t mirroredIdx = (uint16_t)((-leftIdx / n) % 2 == 0 ? (-leftIdx) % n : lastIdx - ((-leftIdx) % n));
                    float32_t mirrored;
                    smemAccessor.get(mirroredIdx, mirrored);
                    result -= mirrored;
                } break;
            }

            color[ch] = result;
        }
        
        texAccessor.set(pixelPos, color * scale);
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

groupshared float32_t smem[2048];

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
    TextureProxy texAccessor;
    SharedMemoryProxy smemAccessor;
    boxBlur(texAccessor, smemAccessor, pc.edgeWrapMode, float32_t4(0, 1, 0, 1), 4, pc.radius, pc.flip);
}
