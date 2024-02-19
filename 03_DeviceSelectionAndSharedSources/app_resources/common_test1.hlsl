struct PSInput
{
    [[vk::location(2)]] nointerpolation uint4 data1 : COLOR1;
    [[vk::location(5)]] float2 data2 : COLOR2;
};

[[vk::binding(6,3)]] RWByteAddressBuffer outputBuff;