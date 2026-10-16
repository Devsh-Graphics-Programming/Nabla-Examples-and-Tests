#include "common.hlsl"

[[vk::push_constant]] SBeautyPushConstants pc;

[numthreads(HashGridWorkgroupSize, HashGridWorkgroupSize, 1)]
[shader("compute")]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	uint32_t linearIdx = ID.y * uint32_t(gSensor.renderSize.x) + ID.x;
    SHashAppendData data = vk::RawBufferLoad<SHashAppendData>(gSensor.pStorageBuffers[SensorUBOBufferAddresses::HashAppendDataBuf] + linearIdx * sizeof(SHashAppendData));

    if (data.isValid > 0u)
    {
        const uint32_t baseIdx = vk::RawBufferLoad<uint32_t>(gSensor.pStorageBuffers[SensorUBOBufferAddresses::IndexBuf] + data.cellIdx * sizeof(uint32_t));
        const uint32_t storeIdx = baseIdx + data.inCellIdx;
        if (storeIdx < HashBufferElementCount)  // TODO: don't know why, but sometimes we end up outside buffer range
            vk::RawBufferStore<uint32_t>(gSensor.pStorageBuffers[SensorUBOBufferAddresses::CellStorageBuf] + storeIdx * sizeof(uint32_t), data.reservoirIdx);
    }
}