#include "common.hlsl"

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/limits.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "nbl/builtin/hlsl/workgroup/arithmetic.hlsl"
#include "nbl/builtin/hlsl/workgroup/scratch_size.hlsl"

using namespace nbl::hlsl;

            // case WRAP_MODE_REPEAT:
            // {
            //     float32_t scanline_last = smemAccessor.get(glsl::gl_WorkGroupSize().x - 1);
            //     float32_t v_floored = ceil((floor(right) - last) / n) * scanline_last + smemAccessor.get(fmod(scanlineIdx + radiusFl - n, n));
            //     float32_t v_ceiled = ceil((ceil(right) - last) / n) * scanline_last + smemAccessor.get(fmod(scanlineIdx + radiusCl - n, n));
            //     sum += lerp(v_floored, v_ceiled, alpha);
            // } break;
            // case WRAP_MODE_MIRROR:
            // {
            //     float32_t scanline_last = smemAccessor.get(glsl::gl_WorkGroupSize().x - 1);
            //     float32_t v_floored;
            //     const uint16_t floored = uint16_t(floor(right));
            //     int16_t d = floored - last;

            //     if (fmod(d, 2 * n) == n)
            //     {
            //         v_floored = ((d + n) / n) * scanline_last;
            //     }
            //     else
            //     {
            //         const uint period = uint(ceil(float(d)/n));

            //         if ((period & 0x1u) == 1)
            //             v_floored = period * scanline_last + scanline_last - smemAccessor.get(glsl::gl_WorkGroupSize().x - uint(fmod(d, n)) - 1);
            //         else
            //             v_floored = period * scanline_last + smemAccessor.get(zeroIdx + fmod(d - 1, n));
            //     }

            //     float32_t v_ceiled;
            //     const uint16_t ceiled = uint16_t(ceil(right));
            //     d = ceiled - last;

            //     if (fmod(d, 2 * n) == n)
            //     {
            //         v_ceiled = ((d + n) / n) * scanline_last;
            //     }
            //     else
            //     {
            //         const uint period = uint(ceil(float(d)/n));

            //         if ((period & 0x1u) == 1)
            //             v_ceiled = period * scanline_last + scanline_last - smemAccessor.get(glsl::gl_WorkGroupSize().x - uint(fmod(d, n)) - 1);
            //         else
            //             v_ceiled = period * scanline_last + smemAccessor.get(zeroIdx + fmod(d - 1, n));
            //     }

            //     sum += lerp(v_floored, v_ceiled, alpha);
            // } break;

            // case WRAP_MODE_REPEAT:
            // {
            //     float32_t scanline_last = smemAccessor.get(glsl::gl_WorkGroupSize().x - 1);
            //     float32_t v_floored = ceil(floor(left) / n) * scanline_last + smemAccessor.get(fmod(scanlineIdx - radiusFl, n));
            //     float32_t v_ceiled = ceil(ceil(left) / n) * scanline_last + smemAccessor.get(fmod(scanlineIdx - radiusCl, n));
            //     sum -= lerp(v_floored, v_ceiled, alpha);
            // } break;
            // case WRAP_MODE_MIRROR:
            // {
            //     float32_t scanline_last = smemAccessor.get(glsl::gl_WorkGroupSize().x - 1);
            //     float32_t v_floored;
            //     const uint16_t floored = uint16_t(floor(left));

            //     if (fmod(abs(floored + 1), 2 * n) == 0)
            //     {
            //         v_floored = -(abs(floored + 1) / n) * scanline_last;
            //     }
            //     else
            //     {
            //         const uint period = uint(ceil(float(abs(floored + 1)) / n));

            //         if ((period & 0x1u) == 1)
            //             v_floored = -1 * (period - 1) * scanline_last - smemAccessor.get(zeroIdx + fmod(abs(floored + 1) - 1, n));
            //         else
            //             v_floored = -1 * (period - 1) * scanline_last - (scanline_last - smemAccessor.get(zeroIdx + fmod(floored + 1, n) - 1));
            //     }

            //     float32_t v_ceiled;
            //     const uint16_t ceiled = uint16_t(ceil(left));

            //     if (ceiled == 0) 
            //     {
            //         v_ceiled = 0;
            //     }
            //     else if (fmod(abs(ceiled + 1), 2 * n) == 0)
            //     {
            //         v_ceiled = -(abs(ceiled + 1) / n) * scanline_last;
            //     }
            //     else
            //     {
            //         const uint period = uint(ceil(float(abs(ceiled + 1)) / n));

            //         if ((period & 0x1u) == 1)
            //             v_ceiled = -1 * (period - 1) * scanline_last - smemAccessor.get(zeroIdx + fmod(abs(ceiled + 1) - 1, n));
            //         else
            //             v_ceiled = -1 * (period - 1) * scanline_last - (scanline_last - smemAccessor.get(zeroIdx + fmod(ceiled + 1, n) - 1));
            //     }

            //     sum -= lerp(v_floored, v_ceiled, alpha);
            // } break;

enum EdgeWrapMode {
	WRAP_MODE_CLAMP_TO_EDGE,
	WRAP_MODE_CLAMP_TO_BORDER,
	WRAP_MODE_REPEAT,
	WRAP_MODE_MIRROR,
};

template<class TextureAccessor, class SharedAccessor, class ScratchAccessor>
void boxBlur(
    NBL_REF_ARG(TextureAccessor) texAccessor,
    NBL_REF_ARG(SharedAccessor) smemAccessor,
    NBL_REF_ARG(ScratchAccessor) scratchAccessor,
    EdgeWrapMode wrapMode,
    float32_t4 borderColor,
    float32_t radius,
    bool flip
) {
    uint16_t2 pixelPos = 0;
    pixelPos[(flip + 1) % 2] = spirv::WorkgroupId.x;

    float32_t scale = 1.0 / (2.0 * radius + 1.0);
    uint16_t radiusFl = (uint16_t)floor(radius);
    uint16_t radiusCl = (uint16_t)ceil(radius);
    float32_t alpha = radius - radiusFl;
    uint16_t n = texAccessor.size()[flip];
    uint16_t lastIdx = n - 1;
    uint16_t itemsPerThread = (uint16_t)ceil((float32_t)n / (float32_t)WORKGROUP_SIZE);

    for (uint16_t i = 0; i < itemsPerThread; i++)
    {
        uint16_t scanIdx = (i * glsl::gl_WorkGroupSize().x) + workgroup::SubgroupContiguousIndex();
        float32_t leftIdx = scanIdx - radius;
        float32_t rightIdx = scanIdx + radius;
        pixelPos[flip] = scanIdx;
        if (pixelPos[flip] > lastIdx) break;

        float32_t4 color = 0;

        for (uint16_t ch = 0; ch < 4; ch++)
        {
            if (spirv::LocalInvocationId.x == 0)
            {
                uint16_t2 pos = pixelPos;
                pos[flip] = 0;
                for (;pos[flip] < n; pos[flip]++)
                    smemAccessor.set(pos[flip], texAccessor.get(pos, ch));
            }

            glsl::barrier();

            // float32_t blurred = 0;
            // smemAccessor.get(scanIdx, blurred);
	        // float32_t sum = 0;
            // sum = workgroup::inclusive_scan<plus<float32_t>, WORKGROUP_SIZE>::template __call<ScratchAccessor>(blurred, scratchAccessor);
            // glsl::barrier();
            // smemAccessor.set(scanIdx, previous_block_sum);
            // glsl::barrier();

            float32_t sum = 0;
            smemAccessor.get(scanIdx, sum);
            for (uint16_t j = 1; j <= radiusFl && scanIdx + j < n; j++)
            {
                float32_t left, right;
                smemAccessor.get(scanIdx - j, left);
                smemAccessor.get(scanIdx + j, right);
                sum += left + right;
            }

            if (rightIdx <= lastIdx)
            {
                float32_t rightFloor, rightCeil;
                smemAccessor.get(scanIdx + radiusFl, rightFloor);
                smemAccessor.get(scanIdx + radiusCl, rightCeil);
                sum += lerp(rightFloor, rightCeil, alpha);
            }
            else switch (wrapMode)
            {
                case WRAP_MODE_CLAMP_TO_EDGE:
                {
                    float32_t last, lastMinus1;
                    smemAccessor.get(lastIdx, last);
                    smemAccessor.get(lastIdx - 1, lastMinus1);
                    sum += (rightIdx - lastIdx) * (last - lastMinus1);
                } break;
                case WRAP_MODE_CLAMP_TO_BORDER:
                {
                    sum += (rightIdx - lastIdx) * borderColor[ch];
                } break;
            }

            if (leftIdx > 0)
            {
                float32_t leftFloor, leftCeil;
                smemAccessor.get(scanIdx - radiusFl, leftFloor);
                smemAccessor.get(scanIdx - radiusCl, leftCeil);
                sum -= lerp(leftFloor, leftCeil, alpha);
            }
            else switch (wrapMode)
            {
                case WRAP_MODE_CLAMP_TO_EDGE:
                {
                    float32_t first;
                    smemAccessor.get(0, first);
                    sum -= abs(leftIdx) * first;
                } break;
                case WRAP_MODE_CLAMP_TO_BORDER:
                {
                    sum -= leftIdx * borderColor[ch];
                } break;
            }

            color[ch] = sum;
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
groupshared uint32_t scratch[scratchSize];

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

struct ScratchProxy
{
	void get(const uint16_t idx, NBL_REF_ARG(float32_t) value)
	{
		value = scratch[idx];
	}

	void set(const uint16_t idx, float32_t value)
	{
        scratch[idx] = value;
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
    ScratchProxy scratchAccessor;
    boxBlur(texAccessor, smemAccessor, scratchAccessor, WRAP_MODE_CLAMP_TO_EDGE, float32_t4(0, 1, 0, 1), 20, pc.flip);
}
