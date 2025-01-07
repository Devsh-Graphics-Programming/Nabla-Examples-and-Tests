#include "common.hlsl"

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/limits.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/workgroup/arithmetic.hlsl"
#include "nbl/builtin/hlsl/workgroup/shared_scan.hlsl"
#include "nbl/builtin/hlsl/workgroup/scratch_size.hlsl"
#include "nbl/builtin/hlsl/colorspace/OETF.hlsl"

using namespace nbl::hlsl;

static const uint16_t WORKGROUP_SIZE = 256;
static const uint16_t PASSES_PER_AXIS = 2;
static const uint16_t arithmeticSz = workgroup::scratch_size_arithmetic<WORKGROUP_SIZE>::value;
static const uint16_t smemSize = WORKGROUP_SIZE + arithmeticSz;
uint32_t3 glsl::gl_WorkGroupSize() { return uint32_t3(WORKGROUP_SIZE, 1, 1); }

[[vk::binding(0)]]
Texture2D<float32_t4> input;
[[vk::binding(1)]]
RWTexture2D<float32_t4> output;

[[vk::push_constant]] PushConstants pc;

[[vk::ext_builtin_input(spv::BuiltInGlobalInvocationId)]]
static const uint32_t3 GlobalInvocationId;

[[vk::ext_builtin_input(spv::BuiltInLocalInvocationIndex)]]
static const uint32_t LocalInvocationIndex;

uint16_t2 byAxis(uint16_t axisIdx, uint16_t val) {
    uint16_t2 r = 0;
    r[axisIdx] = val;
    return r;
}

groupshared float32_t cache[WORKGROUP_SIZE * 2];

struct SharedMemoryAccessor
{
    void get(const uint32_t index, NBL_REF_ARG(float32_t) value)
    {
        value = cache[index];
    }

    void set(const uint32_t index, const float32_t value)
    {
        cache[index] = value;
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        glsl::barrier();
        // GroupMemoryBarrierWithGroupSync();
    }
};


    // float32_t4 color = 0;
    // for (int ch = 0; ch < 4; ch++)
    // {
    //     // if (group_thread_id.x == 0)
    //     // {
    //     //     for (int i = 0; i < WORKGROUP_SIZE; i++)
    //     //     {
    //     //         cache[i] = input[pixel_pos + byAxis(axisIdx, i)][ch];
    //     //     }
    //     // }
        
    //     glsl::barrier();

    //     float sum = input[pixel_pos + byAxis(axisIdx, group_thread_id.x)][ch];
    //     // SharedMemoryAccessor accessor;
    //     // workgroup::inclusive_scan<plus<float32_t>, WORKGROUP_SIZE>::template __call<SharedMemoryAccessor>(sum, accessor);
    //     for (int j = 1; j <= m; j++)
    //     {
    //         sum += input[pixel_pos + byAxis(axisIdx, group_thread_id.x - j)][ch] + input[pixel_pos + byAxis(axisIdx, group_thread_id.x + j)][ch];
    //     }
    //     sum += alpha * (input[pixel_pos + byAxis(axisIdx, group_thread_id.x - m - 1)][ch] + input[pixel_pos + byAxis(axisIdx, group_thread_id.x + m + 1)][ch]);
    //     sum += lerp(input[pixel_pos + byAxis(axisIdx, group_thread_id.x + m+1)][ch], input[pixel_pos + byAxis(axisIdx, group_thread_id.x + m+2)][ch], alpha);
    //     sum -= lerp(input[pixel_pos + byAxis(axisIdx, group_thread_id.x + -m)][ch], input[pixel_pos + byAxis(axisIdx, group_thread_id.x + m-1)][ch], alpha);

    //     color[ch] = sum * scale;
    //     glsl::barrier();
    // }

    // output[pixel_pos] = color;

[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint3 group_thread_id: SV_GroupThreadID, uint3 dispatch_thread_id: SV_DispatchThreadID) {
    uint16_t2 pixel_pos = dispatch_thread_id.xy;
    uint16_t axisIdx = pc.flip;
    if (pc.flip) pixel_pos = pixel_pos.yx;
    float r = 5.0;
    float scale = 1.0f / (2.0f * r + 1.0f);
    int m = (int)r;
    float alpha = r - m;
    SharedMemoryAccessor accessor;

    float32_t4 color = 0;
    for (int ch = 0; ch < 4; ch++)
    {
        if (group_thread_id.x == 0)
        {
            // SharedMemoryAccessor accessor;
            for (int i = 0; i < WORKGROUP_SIZE + m * 2 + 2; i++)
            {
                int2 pos = pixel_pos + byAxis(axisIdx,  i);
                pos[axisIdx] = max(0, pos[axisIdx] - m - 1);
                cache[i] = input[pos][ch];
            }
        }
        
        glsl::barrier();

        float sum = cache[m + 1 + group_thread_id.x];
        // float sum = workgroup::inclusive_scan<plus<float32_t>, WORKGROUP_SIZE>::template __call<SharedMemoryAccessor>(cache[group_thread_id.x], accessor);
        // glsl::barrier();
        // accessor.set(group_thread_id.x, sum);
        // glsl::barrier();
        for (int j = 1; j <= m; j++)
        {
            sum += cache[m + 1 + group_thread_id.x - j] + cache[m + 1 + group_thread_id.x + j];
        }
        sum += alpha * (cache[m + 1 + group_thread_id.x - m - 1] + cache[m + 1 + group_thread_id.x + m + 1]);
        sum += lerp(cache[m + 1 + group_thread_id.x + m+1], cache[m + 1 + group_thread_id.x + m+2], alpha);
        sum -= lerp(cache[m + 1 + group_thread_id.x + -m], cache[m + 1 + group_thread_id.x + m-1], alpha);
        glsl::barrier();
        color[ch] = sum * scale;
    }

    output[pixel_pos] = color;
}
