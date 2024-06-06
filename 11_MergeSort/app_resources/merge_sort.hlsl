#pragma wave shader_stage(compute)

#include "nbl/builtin/hlsl/bda/bda_accessor.hlsl"
#include "app_resources/common.hlsl"

[[vk::push_constant]] MergeSortPushData push_data;

using PtrAccessor = nbl::hlsl::BdaAccessor<int32_t>;

// Rather than indexing into the buffer A / B directly, use group shared memory how much ever possible.
groupshared int shared_memory_input_array[MaxNumberOfArrayElementsSharedMemoryCanHold];

[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint threadIdx : SV_DispatchThreadID)
{
    uint left_array_start = threadIdx * phase_data.num_elements_per_array * 2;
    uint right_array_start = left_array_start + phase_data.num_elements_per_array;

    uint left_array_end = left_array_start + phase_data.num_elements_per_array - 1;
    if (left_array_end >= phase_data.buffer_length)
    {
        left_array_end = phase_data.buffer_length - 1;
    }

    uint right_array_end = right_array_start + phase_data.num_elements_per_array - 1;
    if (right_array_end >= phase_data.buffer_length)
    {
        right_array_end = phase_data.buffer_length - 1;
    }

    // Now that the left and right array bounds are determined, move the data into shared memory.
    // If there is some *excess* data from the buffer A we cannot move to shared memory, store as much as data in shared memory as possible and fetch the rest from global memory.
    for (uint i = left_array_start; i <= min(right_array_end, MaxNumberOfArrayElementsSharedMemoryCanHold - 1); i++)
    {
        shared_memory_input_array[i] = vk::RawBufferLoad<int32_t>(push_data.buffer_a_address + sizeof(int32_t)*i);
    }
    
    GroupMemoryBarrierWithGroupSync();
    
    uint index = left_array_start;

    while (left_array_start <= left_array_end && right_array_start <= right_array_end)
    {
        int buffer_a_data = vk::RawBufferLoad<int32_t>(push_data.buffer_a_address + sizeof(int32_t)*left_array_start);
        int buffer_b_data = vk::RawBufferLoad<int32_t>(push_data.buffer_b_address + sizeof(int32_t)*right_array_start);

        int left_array_current_element = (left_array_start < MaxNumberOfArrayElementsSharedMemoryCanHold) ? shared_memory_input_array[left_array_start] : buffer_a_data;
        int right_array_current_element = (right_array_start < MaxNumberOfArrayElementsSharedMemoryCanHold) ? shared_memory_input_array[right_array_start] : buffer_b_data;

        if (buffer_a_data <= buffer_b_data)
        {
            vk::RawBufferStore<int32_t>(push_data.buffer_b_address+sizeof(int32_t)*index, left_array_current_element);
            ++left_array_start;
            ++index;
        }
        else
        {
            vk::RawBufferStore<int32_t>(push_data.buffer_b_address+sizeof(int32_t)*index, right_array_current_element);
            ++right_array_start;
            ++index;
        }
    }

    while (left_array_start <= left_array_end)
    {
        int buffer_a_data = vk::RawBufferLoad<int32_t>(push_data.buffer_a_address + sizeof(int32_t)*left_array_start);

        int left_array_current_element = (left_array_start < MaxNumberOfArrayElementsSharedMemoryCanHold) ? shared_memory_input_array[left_array_start] : buffer_a_data;
        vk::RawBufferStore<int32_t>(push_data.buffer_b_address+sizeof(int32_t)*index++, left_array_current_element);
        ++left_array_start;
    }

    while (right_array_start <= right_array_end)
    {
        int buffer_b_data = vk::RawBufferLoad<int32_t>(push_data.buffer_b_address + sizeof(int32_t)*right_array_start);

        int right_array_current_element = (right_array_start < MaxNumberOfArrayElementsSharedMemoryCanHold) ? shared_memory_input_array[right_array_start] : buffer_b_data;
        vk::RawBufferStore<int32_t>(push_data.buffer_b_address+sizeof(int32_t)*index++, right_array_current_element);
        ++right_array_start;
    }
}
