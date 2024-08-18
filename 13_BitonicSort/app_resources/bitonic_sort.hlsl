#include "nbl/builtin/hlsl/bda/bda_accessor.hlsl"
#include "app_resources/common.hlsl"

// Define the Push Constant Structure
[[vk::push_constant]] BitonicPushData pushData;

// Define Accessor Types
using Ptr = nbl::hlsl::bda::__ptr < uint32_t >;
using PtrAccessor = nbl::hlsl::BdaAccessor < uint32_t >;

// Define the Group Shared Memory
groupshared uint32_t sdataKey[WorkgroupSize];
groupshared uint32_t sdataValue[WorkgroupSize];

// Define the Shared Accessor for Bitonic Sort
struct SharedAccessor
{
    uint32_t get(const uint32_t index)
    {
        return sdataKey[index];
    }

    void set(const uint32_t index, const uint32_t value)
    {
        sdataKey[index] = value;
    }

    uint32_t getValue(const uint32_t index)
    {
        return sdataValue[index];
    }

    void setValue(const uint32_t index, const uint32_t value)
    {
        sdataValue[index] = value;
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        nbl::hlsl::glsl::barrier();
    }
};

// Bitonic Sort Implementation
void bitonic_sort(uint32_t tid, inout PtrAccessor key, inout PtrAccessor val, inout SharedAccessor sdata, uint32_t size)
{
    // Load data into shared memory
    sdata.set(tid, key.get(tid));

    // Bitonic sort
    for (uint32_t k = 2; k <= size; k <<= 1)
    {
        for (uint32_t j = k >> 1; j > 0; j >>= 1)
        {
            sdata.workgroupExecutionAndMemoryBarrier();
            uint32_t ixj = tid ^ j;

            if (ixj > tid)
            {
                if ((tid & k) == 0)
                {
                    uint32_t left = sdata.get(tid);
                    uint32_t right = sdata.get(ixj);

                    if (left > right)
                    {
                        sdata.set(tid, right);
                        sdata.set(ixj, left);
                    }
                }
                else
                {
                    uint32_t left = sdata.get(ixj);
                    uint32_t right = sdata.get(tid);

                    if (left > right)
                    {
                        sdata.set(ixj, right);
                        sdata.set(tid, left);
                    }
                }
            }
        }
    }

    // Write back sorted data
    sdata.workgroupExecutionAndMemoryBarrier();
    key.set(tid, sdata.get(tid));
}

// Main Shader Function
[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_GroupThreadID, uint32_t3 GroupID : SV_GroupID)
{
    uint32_t tid = ID.x;

    // Define the Accessor Instances
    const Ptr inputKeyPtr = Ptr::create(pushData.inputKeyAddress);
    const Ptr inputValuePtr = Ptr::create(pushData.inputValueAddress);
    const Ptr outputKeyPtr = Ptr::create(pushData.outputKeyAddress);
    const Ptr outputValuePtr = Ptr::create(pushData.outputValueAddress);

    PtrAccessor inputKeyAccessor = PtrAccessor::create(inputKeyPtr);
    PtrAccessor inputValueAccessor = PtrAccessor::create(inputValuePtr);
    PtrAccessor outputKeyAccessor = PtrAccessor::create(outputKeyPtr);
    PtrAccessor outputValueAccessor = PtrAccessor::create(outputValuePtr);

    SharedAccessor sharedAccessor;

    // Calculate thread index
    uint32_t globalID = ID.x + GroupID.x * WorkgroupSize;
    if (globalID < pushData.dataElementCount)
    {
        sharedAccessor.set(ID.x, inputKeyAccessor.get(globalID));
        sharedAccessor.setValue(ID.x, inputValueAccessor.get(globalID));
    }
    sharedAccessor.workgroupExecutionAndMemoryBarrier();

    // Perform Bitonic Sort
    bitonic_sort(tid, inputKeyAccessor, inputValueAccessor, sharedAccessor, pushData.dataElementCount);

    // Store sorted data back to global memory
    sharedAccessor.workgroupExecutionAndMemoryBarrier();
    if (globalID < pushData.dataElementCount)
    {
        outputKeyAccessor.set(globalID, sharedAccessor.get(tid));
        outputValueAccessor.set(globalID, sharedAccessor.getValue(tid));
    }
}
