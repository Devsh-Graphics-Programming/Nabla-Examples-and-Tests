#include "common.hlsl"

// just a small test
#include "nbl/builtin/hlsl/jit/device_capabilities.hlsl"

[[vk::push_constant]] PushConstantData pushConstants;

[numthreads(WorkgroupSize,1,1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
    if (ID.x > pushConstants.maximum - pushConstants.minimum)
        return;

    int count = 0;
    for (uint32_t index = 0; index < pushConstants.dataElementCount; index++)
    {
        if (vk::RawBufferLoad(pushConstants.inputAddress + sizeof(uint32_t) * index) == ID.x)
            count++;
    }

    vk::RawBufferStore(pushConstants.outputAddress + sizeof(uint32_t) * ID.x, count);
}
