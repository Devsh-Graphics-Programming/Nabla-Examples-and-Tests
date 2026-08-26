#include "nbl/builtin/hlsl/bda/bda_accessor.hlsl"
#include "nbl/builtin/hlsl/bda/legacy_bda_accessor.hlsl"

#include "common.hlsl"

[[vk::push_constant]] SBeautyPushConstants pc;

[numthreads(HashGridWorkgroupSize, HashGridWorkgroupSize, 1)]
[shader("compute")]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	uint32_t linearIdx = ID.y * uint32_t(gSensor.renderSize.x) + ID.x;
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
        if (baseIdx + data.inCellIdx < HashBufferElementCount)  // TODO: don't know why this is necessary, but sometimes we end up outside buffer range
            cellStoragePtr.set(baseIdx + data.inCellIdx, data.reservoirIdx);
    }
}