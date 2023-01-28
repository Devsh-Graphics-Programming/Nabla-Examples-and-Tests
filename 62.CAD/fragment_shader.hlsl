#pragma shader_stage(fragment)

enum class ObjectType : uint32_t
{
    LINE = 0u,
    ELLIPSE = 1u,
};

struct DrawObject
{
    ObjectType type;
    uint styleIdx;
    uint64_t address;
};

struct LinePoints
{
    // prev, start, end, next
    double2 p[4u];
};

struct Ellipse
{
    double2 majorAxis;
    double2 center;
    uint2 rangeAnglesPacked; // [0, 2Pi)
    uint eccentricityPacked; // (0, 1]
    uint _pad; // TODO we may need to add prev/next tangent if curve joins are not Bi-Arc
};

struct Globals
{
    double4x4 viewProjection;
    // Next two vars will be part of styles that the objects will reference
    float4 color;
    uint lineWidth;
    uint pad;
    uint2 resolution;
};

struct PSInput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float4 color : COLOR;
    [[vk::location(1)]] nointerpolation float4 start_end : COLOR1;
    [[vk::location(2)]] nointerpolation uint3 lineWidth_eccentricity_objType : COLOR2;
};

[[vk::binding(0, 0)]] ConstantBuffer<Globals> globals : register(b0);
[[vk::binding(1, 0)]] StructuredBuffer<DrawObject> drawObjects : register(t0);

// TODO: move these to files and include them here
//namespace nbl
//{
//    namespace hlsl
//    {
//        namespace shapes
//        {
//            struct Line_t
//            {
//                float2 start, end;
//
//                // https://www.shadertoy.com/view/stcfzn with modifications
//                float getSignedDistance(float2 p, float thickness)
//                {
//                    const float l = length(end - start);
//                    const float2  d = (end - start) / l;
//                    float2  q = p - (start + end) * 0.5;
//                    q = mul(float2x2(d.x, d.y, -d.y, d.x), q);
//                    q = abs(q) - float2(l * 0.5, thickness);
//                    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0);
//                }
//            };
//            struct Circle_t
//            {
//                float2 center;
//                float radius;
//
//                float getSignedDistance(float2 p)
//                {
//                    return distance(p, center) - radius;
//                }
//            };
//
//            struct RoundedLine_t
//            {
//                float2 start, end;
//
//                float getSignedDistance(float2 p, float thickness)
//                {
//                    Circle_t startCircle = { start, thickness };
//                    Circle_t endCircle = { end, thickness };
//                    Line_t mainLine = { start, end };
//                    const float startCircleSD = startCircle.getSignedDistance(p);
//                    const float endCircleSD = endCircle.getSignedDistance(p);
//                    const float lineSD = mainLine.getSignedDistance(p, thickness);
//                    return min(lineSD, min(startCircleSD, endCircleSD));
//                }
//            };
//        }
//    }
//}

namespace SignedDistance
{
    float Line(float2 p, float2 start, float2 end, float lineThickness)
    {
        const float l = length(end - start);
        const float2  d = (end - start) / l;
        float2  q = p - (start + end) * 0.5;
        q = mul(float2x2(d.x, d.y, -d.y, d.x), q);
        q = abs(q) - float2(l * 0.5, lineThickness);
        return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0);
    }

    float Circle(float2 p, float2 center, float radius)
    {
        return distance(p, center) - radius;
    }

    float RoundedLine(float2 p, float2 start, float2 end, float lineThickness)
    {
        const float startCircleSD = Circle(p, start, lineThickness);
        const float endCircleSD = Circle(p, end, lineThickness);
        const float lineSD = Line(p, start, end, lineThickness);
        return min(lineSD, min(startCircleSD, endCircleSD));
    }
}

float4 main(PSInput input) : SV_TARGET
{
    ObjectType objType = (ObjectType)input.lineWidth_eccentricity_objType.z;
    if (objType == ObjectType::ELLIPSE)
    {
    }
    else if (objType == ObjectType::LINE)
    {
        const float2 start = input.start_end.xy;
        const float2 end = input.start_end.zw;
        const float lineThickness = ((float)input.lineWidth_eccentricity_objType.x) / 2.0f;

        float distance = SignedDistance::RoundedLine(input.position.xy, start, end, lineThickness);

        float antiAliasingFactor = /*No need, I'm already in screen space fwidth(distance) * */ 0.5f;
        float alpha = 1.0f - smoothstep(-antiAliasingFactor, +antiAliasingFactor, distance);
        return lerp(float4(0,0,0,0), input.color, alpha);
    }
    return input.color;
}
