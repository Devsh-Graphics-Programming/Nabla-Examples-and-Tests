#include "common.hlsl"

[[vk::binding(0,1)]] RWStructuredBuffer<uint32_t> buff;

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
    buff[ID.x] = ID.x;
}
