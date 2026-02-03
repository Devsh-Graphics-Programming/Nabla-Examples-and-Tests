struct PSInputDebug
{
    float4 position : SV_Position;
};

[shader("pixel")]
float4 fragDebugMain(PSInputDebug input) : SV_TARGET
{
    return float4(1.0, 1.0, 1.0, 1.0);
// return input.color;
}