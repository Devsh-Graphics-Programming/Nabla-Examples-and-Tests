#pragma wave shader_stage(compute)

[[vk::binding(0,0)]] RWStructuredBuffer<int> output_buffer;
[[vk::binding(1,0)]] RWStructuredBuffer<int> input_buffer;

struct SortingPhaseData
{
    uint num_elements_per_array;
    uint buffer_length;
};


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

    uint index = left_array_start;

    while (left_array_start <= left_array_end && right_array_start <= right_array_end)
    {
        if (input_buffer[left_array_start] <= input_buffer[right_array_start])
        {
            output_buffer[index++] = input_buffer[left_array_start++];
        }
        else
        {
            output_buffer[index++] = input_buffer[right_array_start++];
        }
    }

    while (left_array_start <= left_array_end)
    {
        output_buffer[index++] = input_buffer[left_array_start++];
    }

    while (right_array_start <= right_array_end)
    {
        output_buffer[index++] = input_buffer[right_array_start++];
    }
}
