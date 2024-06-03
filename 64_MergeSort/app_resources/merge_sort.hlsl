#pragma wave shader_stage(compute)

[[vk::binding(0,0)]] RWStructuredBuffer<int> output_buffer;
[[vk::binding(1,0)]] RWStructuredBuffer<int> input_buffer;

struct SortingPhaseData
{
    uint num_elements_per_array;
    uint buffer_length;
};

// Rather than indexing into the input array directly, use group shared memory.
groupshared int shared_memory_input_array[MaxNumberOfArrayElementsSharedMemoryCanHold];

[[vk::push_constant]] SortingPhaseData phase_data;

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
    // If there is some *excess* data from the input buffer we cannot move to shared memory, store as much as data in shared memory as possible and fetch the rest from global memory.
    for (uint i = left_array_start; i <= min(right_array_end, MaxNumberOfArrayElementsSharedMemoryCanHold - 1); i++)
    {
        shared_memory_input_array[i] = input_buffer[i];
    }
    
    GroupMemoryBarrierWithGroupSync();
    
    uint index = left_array_start;

    while (left_array_start <= left_array_end && right_array_start <= right_array_end)
    {
        uint left_array_current_element = (left_array_start < MaxNumberOfArrayElementsSharedMemoryCanHold) ? shared_memory_input_array[left_array_start] : input_buffer[left_array_start];
        uint right_array_current_element = (right_array_start < MaxNumberOfArrayElementsSharedMemoryCanHold) ? shared_memory_input_array[right_array_start] : input_buffer[right_array_start];

        
        if (input_buffer[left_array_start] <= input_buffer[right_array_start])
        {
            output_buffer[index++] = left_array_current_element;
            ++left_array_start;
        }
        else
        {
            output_buffer[index++] = right_array_current_element;
            ++right_array_start;
        }
    }

    while (left_array_start <= left_array_end)
    {
        int left_array_current_element = (left_array_start < MaxNumberOfArrayElementsSharedMemoryCanHold) ? shared_memory_input_array[left_array_start] : input_buffer[left_array_start];
        output_buffer[index++] = left_array_current_element;
        ++left_array_start;
    }

    while (right_array_start <= right_array_end)
    {
        int right_array_current_element = (right_array_start < MaxNumberOfArrayElementsSharedMemoryCanHold) ? shared_memory_input_array[right_array_start] : input_buffer[right_array_start];
        output_buffer[index++] = right_array_current_element;
        ++right_array_start;
    }
}
