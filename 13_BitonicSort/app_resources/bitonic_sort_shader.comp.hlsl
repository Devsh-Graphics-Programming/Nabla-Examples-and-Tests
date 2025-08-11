#include "nbl/builtin/hlsl/bda/bda_accessor.hlsl"

struct BitonicPushData
{
    uint64_t inputKeyAddress;
    uint64_t inputValueAddress;
    uint64_t outputKeyAddress;
    uint64_t outputValueAddress;
    uint32_t dataElementCount;
};

using namespace nbl::hlsl;

[[vk::push_constant]] BitonicPushData pushData;

using DataPtr = bda::__ptr<uint32_t>;
using DataAccessor = BdaAccessor<uint32_t>;

groupshared uint32_t sharedKeys[ElementCount];
groupshared uint32_t sharedValues[ElementCount];

[numthreads(WorkgroupSize, 1, 1)]
[shader("compute")]
void main(uint32_t3 dispatchId : SV_DispatchThreadID, uint32_t3 localId : SV_GroupThreadID)
{
    const uint32_t threadId = localId.x;
    const uint32_t dataSize = pushData.dataElementCount;
    
    DataAccessor inputKeys = DataAccessor::create(DataPtr::create(pushData.inputKeyAddress));
    DataAccessor inputValues = DataAccessor::create(DataPtr::create(pushData.inputValueAddress));
    
    for (uint32_t i = threadId; i < dataSize; i += WorkgroupSize)
    {
        inputKeys.get(i, sharedKeys[i]);
        inputValues.get(i, sharedValues[i]);
    }
    
    // Synchronize all threads after loading
    GroupMemoryBarrierWithGroupSync();
    

    for (uint32_t stage = 0; stage < Log2ElementCount; stage++)
    {
        for (uint32_t pass = 0; pass <= stage; pass++)
        {
            const uint32_t compareDistance = 1 << (stage - pass);
            
            for (uint32_t i = threadId; i < dataSize; i += WorkgroupSize)
            {
                const uint32_t partnerId = i ^ compareDistance;
                
                if (partnerId >= dataSize)
                    continue;
               
                const uint32_t waveSize = WaveGetLaneCount();
                const uint32_t myWaveId = i / waveSize;
                const uint32_t partnerWaveId = partnerId / waveSize;
                const bool sameWave = (myWaveId == partnerWaveId);

                uint32_t myKey, myValue, partnerKey, partnerValue;
                [branch]
                if (sameWave && compareDistance < waveSize)
                {
                    // WAVE INTRINSIC
                    myKey = sharedKeys[i];
                    myValue = sharedValues[i];

                    const uint32_t partnerLane = partnerId % waveSize;
                    partnerKey = WaveReadLaneAt(myKey, partnerLane);
                    partnerValue = WaveReadLaneAt(myValue, partnerLane);
                }
                else
                {
                    // SHARED MEM
                    myKey = sharedKeys[i];
                    myValue = sharedValues[i];
                    partnerKey = sharedKeys[partnerId];
                    partnerValue = sharedValues[partnerId];
                }

                const uint32_t sequenceSize = 1 << (stage + 1);
                const uint32_t sequenceIndex = i / sequenceSize;
                const bool sequenceAscending = (sequenceIndex % 2) == 0;
                const bool ascending = true;
                const bool finalDirection = sequenceAscending == ascending;
                
                const bool swap = (myKey > partnerKey) == finalDirection;
                
                // WORKGROUP COORDINATION: Only lower-indexed element writes both
                if (i < partnerId && swap)
                {
                    sharedKeys[i] = partnerKey;
                    sharedKeys[partnerId] = myKey;
                    sharedValues[i] = partnerValue;
                    sharedValues[partnerId] = myValue;
                }
            }
            
            GroupMemoryBarrierWithGroupSync();
        }
    }
    

    DataAccessor outputKeys = DataAccessor::create(DataPtr::create(pushData.outputKeyAddress));
    DataAccessor outputValues = DataAccessor::create(DataPtr::create(pushData.outputValueAddress));
    
    for (uint32_t i = threadId; i < dataSize; i += WorkgroupSize)
    {
        outputKeys.set(i, sharedKeys[i]);
        outputValues.set(i, sharedValues[i]);
    }
}