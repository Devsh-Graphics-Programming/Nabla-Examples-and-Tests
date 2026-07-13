#include "common.hlsl"
#include "renderer/shaders/bda_accessors.hlsl"

NBL_CONSTEXPR uint32_t WorkgroupSize = WORKGROUP_SIZE;

[[vk::push_constant]] SBeautyPushConstants pc;

[numthreads(WorkgroupSize, WorkgroupSize, 1)]
[shader("compute")]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	uint linearIdx = pixel.y * params.frameDim.x + pixel.x;
    SHashAppendData data;
    LegacyBdaAccessor<SHashAppendData> hashAppendDataPtr = LegacyBdaAccessor<SHashAppendData>::create(gSensor.pStorageBuffers[SensorUBOBufferAddresses::HashAppendDataBuf]);
    hashAppendDataPtr.get(linearIdx, data);

    if (data.isValid)
    {
        bda::__ptr<uint32_t> ptr0 = bda::__ptr<uint32_t>::create(gSensor.pStorageBuffers[SensorUBOBufferAddresses::IndexBuf]);
        BdaAccessor<uint32_t> indexPtr = BdaAccessor<uint32_t>::create(ptr0);
        uint32_t baseIdx;
        indexPtr.get(data.cellIdx, baseIdx);
        bda::__ptr<uint32_t> ptr1 = bda::__ptr<uint32_t>::create(gSensor.pStorageBuffers[SensorUBOBufferAddresses::CellStorageBuf]);
        BdaAccessor<uint32_t> cellStoragePtr = BdaAccessor<uint32_t>::create(ptr1);
        cellStoragePtr.set(baseIdx + data.inCellIdx, data.reservoirIdx);
    }
}