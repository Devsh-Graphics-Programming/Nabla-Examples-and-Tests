#pragma wave shader_stage(compute)

layout(local_size_x = 16, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) uint32_t SPEC_CONSTANT_VALUE = 10;
layout(binding = 0, set = 0) buffer OutputBuff
{
    int data[];
}

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
    uint index = gl_GlobalInvocationID.x;
    data[index] = SPEC_CONSTANT_VALUE;
}