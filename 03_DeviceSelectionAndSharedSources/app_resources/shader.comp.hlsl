// intentionally making my live difficult here, to showcase the power of reflection
[[vk::binding(2,3)]] ByteAddressBuffer inputs[2];
[[vk::binding(6,3)]] RWByteAddressBuffer output;

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	buff[ID.x] = ID.x;
}