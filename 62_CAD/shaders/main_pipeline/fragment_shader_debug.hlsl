
#pragma shader_stage(fragment)

struct PSInput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float4 color : COLOR;
    [[vk::location(1)]] nointerpolation float4 start_end : COLOR1;
    [[vk::location(2)]] nointerpolation uint3 lineWidth_eccentricity_objType : COLOR2;
};

float4 main(PSInput input) : SV_TARGET
{
    return float4(1.0, 1.0, 1.0, 1.0);
// return input.color;
}