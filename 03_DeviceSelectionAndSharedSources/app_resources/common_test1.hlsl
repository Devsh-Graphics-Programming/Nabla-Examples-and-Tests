struct PSInput
{
    [[vk::location(0)]] nointerpolation uint4 data1 : COLOR1;
    [[vk::location(1)]] float4 data2 : COLOR2;
};