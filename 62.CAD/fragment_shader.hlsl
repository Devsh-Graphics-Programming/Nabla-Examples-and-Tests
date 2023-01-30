#pragma shader_stage(fragment)

#include "common.hlsl"

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
//                static Line_t create(float2 start, float2 end)
//                {
//                    Line_t ret;
//                    ret.start = start;
//                    ret.end = end;
//                    return ret;
//                }
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

    float msign(in float x) { return (x < 0.0) ? -1.0 : 1.0; }

    // https://iquilezles.org/articles/ellipsedist/ with modifications to add rotation and different inputs
    float Ellipse(float2 p, float2 center, float2 majorAxis, float eccentricity)
    {
        float majorAxisLength = length(majorAxis);
        if (eccentricity == 1.0f) return length(p - center) - majorAxisLength;

        float minorAxisLength = majorAxisLength * eccentricity;
        float2 ab = float2(majorAxisLength, minorAxisLength);

        float2 dir = majorAxis / majorAxisLength;
        p = p - center;
        p = abs(mul(float2x2(dir.x, dir.y, -dir.y, dir.x), p));

        if (p.x > p.y) { p = p.yx; ab = ab.yx; }

        float l = ab.y * ab.y - ab.x * ab.x;

        float m = ab.x * p.x / l;
        float n = ab.y * p.y / l;
        float m2 = m * m;
        float n2 = n * n;

        float c = (m2 + n2 - 1.0) / 3.0;
        float c3 = c * c * c;

        float d = c3 + m2 * n2;
        float q = d + m2 * n2;
        float g = m + m * n2;

        float co;

        if (d < 0.0)
        {
            float h = acos(q / c3) / 3.0;
            float s = cos(h) + 2.0;
            float t = sin(h) * sqrt(3.0);
            float rx = sqrt(m2 - c * (s + t));
            float ry = sqrt(m2 - c * (s - t));
            co = ry + sign(l) * rx + abs(g) / (rx * ry);
        }
        else
        {
            float h = 2.0 * m * n * sqrt(d);
            float s = msign(q + h) * pow(abs(q + h), 1.0 / 3.0);
            float t = msign(q - h) * pow(abs(q - h), 1.0 / 3.0);
            float rx = -(s + t) - c * 4.0 + 2.0 * m2;
            float ry = (s - t) * sqrt(3.0);
            float rm = sqrt(rx * rx + ry * ry);
            co = ry / sqrt(rm - rx) + 2.0 * g / rm;
        }
        co = (co - m) / 2.0;

        float si = sqrt(max(1.0 - co * co, 0.0));

        float2 r = ab * float2(co, si);

        return length(r - p) * msign(p.y - r.y);
    }

    float EllipseOutline(float2 p, float2 center, float2 majorAxis, float eccentricity, float thickness)
    {
        float ellipseDist = Ellipse(p, center, majorAxis, eccentricity);
        return abs(ellipseDist) - thickness;
    }
}

float4 main(PSInput input) : SV_TARGET
{
    ObjectType objType = (ObjectType)asuint(input.lineWidth_eccentricity_objType.z);

    if (objType == ObjectType::ELLIPSE)
    {
        return float4(1.0, 0.0, 1.0, 1.0);

        const float2 center = input.start_end.x;
        const float2 majorAxis = input.start_end.y;
        const float lineThickness = input.lineWidth_eccentricity_objType.x / 2.0f;
        const float eccentricity = asuint(input.lineWidth_eccentricity_objType.y) /*TODO: unpack into double*/;

        float distance = SignedDistance::EllipseOutline(input.position.xy, center, majorAxis, eccentricity, lineThickness);

        const float antiAliasingFactor = globals.antiAliasingFactor;
        float alpha = 1.0f - smoothstep(-antiAliasingFactor, +antiAliasingFactor, distance);
        return float4(input.color.xyz, alpha);
    }
    else if (objType == ObjectType::LINE)
    {
        const float2 start = input.start_end.xy;
        const float2 end = input.start_end.zw;
        const float lineThickness = input.lineWidth_eccentricity_objType.x / 2.0f;
        float distance = SignedDistance::RoundedLine(input.position.xy, start, end, lineThickness);
        /* No need to mul with fwidth(distance), distance already in screen space */
        const float antiAliasingFactor = globals.antiAliasingFactor;
        float alpha = 1.0f - smoothstep(-antiAliasingFactor, +antiAliasingFactor, distance);
        return float4(input.color.xyz, alpha);
    }


    return input.color;
}
