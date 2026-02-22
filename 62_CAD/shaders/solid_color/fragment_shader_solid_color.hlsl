// TODO: extract into common.hlsl, make sure to differentiate with other push constants of other pipelines (use namespaces?!)
struct PushConstants
{
    float4 solidColor;
};

[[vk::push_constant]] PushConstants pc;

[shader("pixel")]
float4 fragSolidColor(float4 position : SV_Position) : SV_TARGET
{
    return pc.solidColor;
}
