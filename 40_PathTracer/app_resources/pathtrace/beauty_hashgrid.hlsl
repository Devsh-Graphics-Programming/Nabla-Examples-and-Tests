#include "nbl/builtin/hlsl/bda/bda_accessor.hlsl"
#include "nbl/builtin/hlsl/bda/legacy_bda_accessor.hlsl"

#include "common.hlsl"

NBL_CONSTEXPR uint32_t WorkgroupSize = 16u;

[[vk::push_constant]] SBeautyPushConstants pc;

[numthreads(WorkgroupSize, WorkgroupSize, 1)]
[shader("compute")]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	uint linearIdx = ID.y * gSensor.renderSize.x + ID.x;
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