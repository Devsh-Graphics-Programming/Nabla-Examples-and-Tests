
struct VertexOut {
    float32_t4 ndc : SV_Position;
    float32_t3 meta : COLOR1;
};


[shader("pixel")]
float32_t4 main(VertexOut input) : SV_Target0
{
    const float32_t3 normal = input.meta;
    return float32_t4(normalize(normal) * 0.5f + float32_t3(0.5f, 0.5f, 0.5f), 1.f);
}