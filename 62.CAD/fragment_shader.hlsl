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
    float Ellipse(float2 p, float2 center, float2 majorAxis, double eccentricity)
    {
        float majorAxisLength = length(majorAxis);

        if (eccentricity == 1.0)
            return length(p - center) - majorAxisLength;

        double minorAxisLength = double(majorAxisLength * eccentricity);
        double2 ab = double2(majorAxisLength, minorAxisLength);

        double2 dir = majorAxis / majorAxisLength;
        p = p - center;
        p = abs(mul(double2x2(dir.x, dir.y, -dir.y, dir.x), p));

        if (p.x > p.y) { p = p.yx; ab = ab.yx; }

        double l = ab.y * ab.y - ab.x * ab.x;

        double m = ab.x * p.x / l;
        double n = ab.y * p.y / l;
        double m2 = m * m;
        double n2 = n * n;

        double c = (m2 + n2 - 1.0) / 3.0;
        double c3 = c * c * c;

        double d = c3 + m2 * n2;
        double q = d + m2 * n2;
        double g = m + m * n2;

        double co;

        if (d < 0.0)
        {
            double h = acos(q / c3) / 3.0;
            double s = cos(h) + 2.0;
            double t = sin(h) * sqrt(3.0);
            double rx = sqrt(m2 - c * (s + t));
            double ry = sqrt(m2 - c * (s - t));
            co = ry + sign(l) * rx + abs(g) / (rx * ry);
        }
        else
        {
            double h = 2.0 * m * n * sqrt(d);
            double s = msign(q + h) * pow(abs(q + h), 1.0 / 3.0);
            double t = msign(q - h) * pow(abs(q - h), 1.0 / 3.0);
            double rx = -(s + t) - c * 4.0 + 2.0 * m2;
            double ry = (s - t) * sqrt(3.0);
            double rm = sqrt(rx * rx + ry * ry);
            co = ry / sqrt(rm - rx) + 2.0 * g / rm;
        }
        co = (co - m) / 2.0;

        double si = sqrt(max(1.0 - co * co, 0.0));

        double2 r = ab * double2(co, si);

        return length(r - p) * msign(p.y - r.y);
    }

    float EllipseOutline(float2 p, float2 center, float2 majorAxis, double eccentricity, float thickness)
    {
        float ellipseDist = Ellipse(p, center, majorAxis, eccentricity);
        return abs(ellipseDist) - thickness;
    }
}

#if defined(NBL_FEATURE_FRAGMENT_SHADER_PIXEL_INTERLOCK)
[[vk::ext_instruction(/* OpBeginInvocationInterlockEXT */ 5364)]]
void beginInvocationInterlockEXT();
[[vk::ext_instruction(/* OpEndInvocationInterlockEXT */ 5365)]]
void endInvocationInterlockEXT();
#endif

float4 main(PSInput input) : SV_TARGET
{
#if defined(NBL_FEATURE_FRAGMENT_SHADER_PIXEL_INTERLOCK)
    [[vk::ext_capability(/*FragmentShaderPixelInterlockEXT*/ 5378)]]
    [[vk::ext_extension("SPV_EXT_fragment_shader_interlock")]]
    vk::ext_execution_mode(/*PixelInterlockOrderedEXT*/ 5366);
#endif

    ObjectType objType = (ObjectType)input.lineWidth_eccentricity_objType_writeToAlpha.z;

    float localAlpha = 0.0f;
    bool writeToAlpha = input.lineWidth_eccentricity_objType_writeToAlpha.w == 1u;
    
    if (writeToAlpha)
    {
        if (objType == ObjectType::ELLIPSE)
        {
            const float2 center = input.start_end.xy;
            const float2 majorAxis = input.start_end.zw;
            const float lineThickness = asfloat(input.lineWidth_eccentricity_objType_writeToAlpha.x) / 2.0f;
            const double eccentricity = (double)(input.lineWidth_eccentricity_objType_writeToAlpha.y) / UINT32_MAX;

            float distance = SignedDistance::EllipseOutline(input.position.xy, center, majorAxis, eccentricity, lineThickness);

            const float antiAliasingFactor = globals.antiAliasingFactor;
            localAlpha = 1.0f - smoothstep(-antiAliasingFactor, +antiAliasingFactor, distance);
        }
        else if (objType == ObjectType::LINE)
        {
            const float2 start = input.start_end.xy;
            const float2 end = input.start_end.zw;
            const float lineThickness = asfloat(input.lineWidth_eccentricity_objType_writeToAlpha.x) / 2.0f;
            float distance = SignedDistance::RoundedLine(input.position.xy, start, end, lineThickness);

            /* No need to mul with fwidth(distance), distance already in screen space */
            const float antiAliasingFactor = globals.antiAliasingFactor;
            localAlpha = 1.0f - smoothstep(-antiAliasingFactor, +antiAliasingFactor, distance);
        }
        else if (objType == ObjectType::ROAD)
        {
            localAlpha = 1.0f;
            // use fwidth to get antiAliasingFactor because road line width is constant in world space
            // calculate alpha based on aniAliasingFactor
        }
    }

    uint2 fragCoord = uint2(input.position.xy);

    float alpha = 0.0f; // new alpha

#if defined(NBL_FEATURE_FRAGMENT_SHADER_PIXEL_INTERLOCK)
    beginInvocationInterlockEXT();

    alpha = asfloat(pseudoStencil[fragCoord]);
    if (writeToAlpha)
    {
        if (localAlpha > alpha)
            pseudoStencil[fragCoord] = asuint(localAlpha);
    }
    else
    {
        if (alpha != 0.0f)
            pseudoStencil[fragCoord] = asuint(0.0f);
    }

    endInvocationInterlockEXT();

    if (writeToAlpha || alpha == 0.0f)
        discard;
#else
    alpha = localAlpha;
    if (!writeToAlpha)
        discard;
    //if (writeToAlpha)
    //{
    //    InterlockedMax(pseudoStencil[fragCoord], asuint(localAlpha));
    //}
    //else
    //{
    //    uint previousAlpha;
    //    InterlockedExchange(pseudoStencil[fragCoord], 0u, previousAlpha);
    //    alpha = previousAlpha;
    //}
#endif

    return float4(input.color.xyz, input.color.w * alpha);
}
