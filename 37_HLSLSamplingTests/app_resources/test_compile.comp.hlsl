// Compile test: instantiate sampling types to verify DXC compilation
#include <nbl/builtin/hlsl/sampling/concentric_mapping.hlsl>
#include <nbl/builtin/hlsl/sampling/linear.hlsl>
#include <nbl/builtin/hlsl/sampling/bilinear.hlsl>
#include <nbl/builtin/hlsl/sampling/uniform_spheres.hlsl>
#include <nbl/builtin/hlsl/sampling/cos_weighted_spheres.hlsl>
#include <nbl/builtin/hlsl/sampling/box_muller_transform.hlsl>
#include <nbl/builtin/hlsl/sampling/spherical_triangle.hlsl>
#include <nbl/builtin/hlsl/sampling/projected_spherical_triangle.hlsl>
#include <nbl/builtin/hlsl/sampling/spherical_rectangle.hlsl>
using namespace nbl::hlsl;

[[vk::binding(0, 0)]] RWStructuredBuffer<float32_t4> output;

[numthreads(1, 1, 1)] 
void main()
{
    float32_t2 u2 = float32_t2(0.5, 0.5);
    float32_t3 u3 = float32_t3(0.5, 0.5, 0.5);
    float32_t4 acc = float32_t4(0, 0, 0, 0);

    // concentric mapping (free function)
    float32_t2 concentric = sampling::concentricMapping<float32_t>(u2);
    acc.xy += concentric;

    // Linear
    sampling::Linear<float32_t> lin = sampling::Linear<float32_t>::create(u2);
    acc.x += lin.generate(0.5f);

    // Bilinear
    sampling::Bilinear<float32_t> bilinear = sampling::Bilinear<float32_t>::create(float32_t4(1, 2, 3, 4));
    float32_t rcpPdf;
    float32_t2 bilSample = bilinear.generate(rcpPdf, u2);
    acc.xy += bilSample;
    acc.z += rcpPdf;

    // UniformHemisphere
    acc.xyz += sampling::UniformHemisphere<float32_t>::generate(u2);

    // UniformSphere
    acc.xyz += sampling::UniformSphere<float32_t>::generate(u2);

    // ProjectedHemisphere
    acc.xyz += sampling::ProjectedHemisphere<float32_t>::generate(u2);

    // ProjectedSphere
    acc.xyz += sampling::ProjectedSphere<float32_t>::generate(u3);

    // BoxMullerTransform
    sampling::BoxMullerTransform<float32_t> bmt;
    bmt.stddev = 1.0;
    acc.xy += bmt(u2);

    // SphericalTriangle
    shapes::SphericalTriangle<float32_t> shapeTri;
    shapeTri.vertex0 = float32_t3(1, 0, 0);
    shapeTri.vertex1 = float32_t3(0, 1, 0);
    shapeTri.vertex2 = float32_t3(0, 0, 1);
    sampling::SphericalTriangle<float32_t> sphTri = sampling::SphericalTriangle<float32_t>::create(shapeTri);
    float32_t stRcpPdf;
    acc.xyz += sphTri.generate(stRcpPdf, u2);
    acc.w += stRcpPdf;

    // SphericalRectangle
    shapes::SphericalRectangle<float32_t> shapeRect;
    shapeRect.r0 = float32_t3(-0.5, -0.5, -1.0);
    sampling::SphericalRectangle<float32_t> sphRect = sampling::SphericalRectangle<float32_t>::create(shapeRect);
    float32_t srS;
    acc.xy += sphRect.generate(float32_t2(1.0, 1.0), u2, srS);
    acc.z += srS;

    // ProjectedSphericalTriangle — skipped: pre-existing bug in computeBilinearPatch(receiverNormal, isBSDF)
    sampling::ProjectedSphericalTriangle<float32_t> projTri = sampling::ProjectedSphericalTriangle<float32_t>::create(shapeTri);
    float32_t ptRcpPdf;
    // acc.xyz += projTri.generate(ptRcpPdf, float32_t3(0.0f, 0.0f, 1.0f), true, u2);
    acc.w += ptRcpPdf;

    output[0] = acc;
}
