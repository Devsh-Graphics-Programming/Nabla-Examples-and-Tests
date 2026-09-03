// Compile test: instantiate all sampling types and their concept-required methods to verify DXC compilation
#include <nbl/builtin/hlsl/sampling/concentric_mapping.hlsl>
#include <nbl/builtin/hlsl/sampling/polar_mapping.hlsl>
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
[shader("compute")] 
void main()
{
   float32_t2 u2 = float32_t2(0.5, 0.5);
   float32_t3 u3 = float32_t3(0.5, 0.5, 0.5);
   float32_t4 acc = float32_t4(0, 0, 0, 0);

   // ConcentricMapping — generate, generateInverse, forwardPdf, backwardPdf, forwardWeight, backwardWeight
   sampling::ConcentricMapping<float32_t>::cache_type cache;
   float32_t2 concentric = sampling::ConcentricMapping<float32_t>::generate(u2, cache);
   acc.xy += concentric;
   acc.xy += sampling::ConcentricMapping<float32_t>::generateInverse(concentric);
   acc.x += sampling::ConcentricMapping<float32_t>::forwardPdf(u2, cache);
   acc.x += sampling::ConcentricMapping<float32_t>::backwardPdf(concentric);
   acc.x += sampling::ConcentricMapping<float32_t>::forwardWeight(u2, cache);
   acc.x += sampling::ConcentricMapping<float32_t>::backwardWeight(concentric);

   // PolarMapping — generate, generateInverse, forwardPdf, backwardPdf, forwardWeight, backwardWeight
   sampling::PolarMapping<float32_t>::cache_type polarCache;
   float32_t2 polar = sampling::PolarMapping<float32_t>::generate(u2, polarCache);
   acc.xy += polar;
   acc.xy += sampling::PolarMapping<float32_t>::generateInverse(polar);
   acc.x += sampling::PolarMapping<float32_t>::forwardPdf(u2, polarCache);
   acc.x += sampling::PolarMapping<float32_t>::backwardPdf(polar);
   acc.x += sampling::PolarMapping<float32_t>::forwardWeight(u2, polarCache);
   acc.x += sampling::PolarMapping<float32_t>::backwardWeight(polar);

   // Linear — generate, generateInverse, forwardPdf, backwardPdf, forwardWeight, backwardWeight
   sampling::Linear<float32_t> lin = sampling::Linear<float32_t>::create(u2);
   sampling::Linear<float32_t>::cache_type linCache;
   float32_t linSample = lin.generate(0.5f, linCache);
   acc.x += linSample;
   acc.x += lin.forwardPdf(0.5f, linCache);
   acc.x += lin.forwardWeight(0.5f, linCache);
   acc.x += lin.backwardPdf(linSample);
   acc.x += lin.backwardWeight(linSample);

   // Bilinear — generate, generateInverse, forwardPdf, backwardPdf, forwardWeight, backwardWeight
   sampling::Bilinear<float32_t> bilinear = sampling::Bilinear<float32_t>::create(float32_t4(1, 2, 3, 4));
   sampling::Bilinear<float32_t>::cache_type bilCache;
   float32_t2 bilSample = bilinear.generate(u2, bilCache);
   acc.xy += bilSample;
   acc.x += bilinear.forwardPdf(u2, bilCache);
   acc.x += bilinear.forwardWeight(u2, bilCache);
   acc.x += bilinear.backwardPdf(bilSample);
   acc.x += bilinear.backwardWeight(bilSample);

   // UniformHemisphere — generate, generateInverse, forwardPdf, backwardPdf, forwardWeight, backwardWeight
   sampling::UniformHemisphere<float32_t> uniHemi;
   sampling::UniformHemisphere<float32_t>::cache_type uniHemiCache;
   float32_t3 uniHemiSample = uniHemi.generate(u2, uniHemiCache);
   acc.xyz += uniHemiSample;
   acc.x += uniHemi.forwardPdf(u2, uniHemiCache);
   acc.x += uniHemi.forwardWeight(u2, uniHemiCache);
   acc.xy += uniHemi.generateInverse(uniHemiSample);
   acc.x += uniHemi.backwardPdf(uniHemiSample);
   acc.x += uniHemi.backwardWeight(uniHemiSample);

   // UniformSphere — generate, generateInverse, forwardPdf, backwardPdf, forwardWeight, backwardWeight
   sampling::UniformSphere<float32_t> uniSph;
   sampling::UniformSphere<float32_t>::cache_type uniSphCache;
   float32_t3 uniSphSample = uniSph.generate(u2, uniSphCache);
   acc.xyz += uniSphSample;
   acc.x += uniSph.forwardPdf(u2, uniSphCache);
   acc.x += uniSph.forwardWeight(u2, uniSphCache);
   acc.xy += uniSph.generateInverse(uniSphSample);
   acc.x += uniSph.backwardPdf(uniSphSample);
   acc.x += uniSph.backwardWeight(uniSphSample);

   // ProjectedHemisphere — generate, generateInverse, forwardPdf, backwardPdf, forwardWeight, backwardWeight
   sampling::ProjectedHemisphere<float32_t>::cache_type projHemiCache;
   float32_t3 projHemi = sampling::ProjectedHemisphere<float32_t>::generate(u2, projHemiCache);
   acc.xyz += projHemi;
   acc.x += sampling::ProjectedHemisphere<float32_t>::forwardPdf(u2, projHemiCache);
   acc.x += sampling::ProjectedHemisphere<float32_t>::forwardWeight(u2, projHemiCache);
   acc.xy += sampling::ProjectedHemisphere<float32_t>::generateInverse(projHemi);
   acc.x += sampling::ProjectedHemisphere<float32_t>::backwardPdf(projHemi);
   acc.x += sampling::ProjectedHemisphere<float32_t>::backwardWeight(projHemi);

   // ProjectedSphere — generate, generateInverse, forwardPdf, backwardPdf, forwardWeight, backwardWeight
   sampling::ProjectedSphere<float32_t> projSphSampler;
   sampling::ProjectedSphere<float32_t>::cache_type projSphCache;
   float32_t3 projSphereSample = u3;
   float32_t3 projSphere = projSphSampler.generate(projSphereSample, projSphCache);
   acc.xyz += projSphere;
   acc.x += projSphSampler.forwardPdf(projSphereSample, projSphCache);
   acc.x += projSphSampler.forwardWeight(projSphereSample, projSphCache);
   acc.x += projSphSampler.backwardPdf(projSphere);
   acc.x += projSphSampler.backwardWeight(projSphere);

   // BoxMullerTransform — generate, forwardPdf, backwardPdf, forwardWeight, backwardWeight
   sampling::BoxMullerTransform<float32_t> bmt = sampling::BoxMullerTransform<float32_t>::create(1.0);
   sampling::BoxMullerTransform<float32_t>::cache_type bmtCache;
   float32_t2 bmtSample = bmt.generate(u2, bmtCache);
   acc.xy += bmtSample;
   acc.x += bmt.forwardPdf(u2, bmtCache);
   acc.x += bmt.forwardWeight(u2, bmtCache);
   acc.x += bmt.backwardPdf(bmtSample);
   acc.x += bmt.backwardWeight(bmtSample);
   acc.xy += bmt.separateBackwardPdf(bmtSample);

   // SphericalTriangle — generate, generateInverse, forwardPdf, backwardPdf, forwardWeight, backwardWeight
   // Octant triangle: all dot products between vertices are 0, so cos_sides=0, csc_sides=1
   const float32_t3 triVerts[3] = {float32_t3(1, 0, 0), float32_t3(0, 1, 0), float32_t3(0, 0, 1)};
   shapes::SphericalTriangle<float32_t> shapeTri = shapes::SphericalTriangle<float32_t>::createFromUnitSphereVertices(triVerts);
   sampling::SphericalTriangle<float32_t, true> sphTri = sampling::SphericalTriangle<float32_t, true>::create(shapeTri);
   sampling::SphericalTriangle<float32_t>::cache_type sphTriCache;
   float32_t3 stSample = sphTri.generate(u2, sphTriCache);
   acc.xyz += stSample;
   acc.x += sphTri.forwardPdf(u2, sphTriCache);
   acc.x += sphTri.forwardWeight(u2, sphTriCache);
   acc.xy += sphTri.generateInverse(stSample);
   acc.x += sphTri.backwardPdf(stSample);
   acc.x += sphTri.backwardWeight(stSample);

   // SphericalRectangle — generate, forwardPdf, backwardPdf, forwardWeight, backwardWeight
   shapes::CompressedSphericalRectangle<float32_t> csr;
   csr.origin = float32_t3(0.0, 0.0, -1.0);
   csr.right = float32_t3(1.0, 0.0, 0.0);
   csr.up = float32_t3(0.0, 1.0, 0.0);
   shapes::SphericalRectangle<float32_t> shapeRect = shapes::SphericalRectangle<float32_t>::create(csr);
   const float32_t3 srObserver = float32_t3(0.0, 0.0, 0.0);
   sampling::SphericalRectangle<float32_t> sphRect = sampling::SphericalRectangle<float32_t>::create(shapeRect, srObserver);
   sampling::SphericalRectangle<float32_t>::cache_type sphRectCache;
   float32_t3 srSample = sphRect.generate(u2, sphRectCache);
   acc.xyz += srSample;
   acc.x += sphRect.forwardPdf(u2, sphRectCache);
   acc.x += sphRect.forwardWeight(u2, sphRectCache);
   acc.x += sphRect.backwardPdf(srSample);
   acc.x += sphRect.backwardWeight(srSample);

   // ProjectedSphericalTriangle — generate, forwardPdf, backwardPdf, forwardWeight, backwardWeight
   sampling::ProjectedSphericalTriangle<float32_t> projTri = sampling::ProjectedSphericalTriangle<float32_t>::create(shapeTri, float32_t3(0.0, 0.0, 1.0), false);
   sampling::ProjectedSphericalTriangle<float32_t>::cache_type projTriCache;
   float32_t3 ptSample = projTri.generate(u2, projTriCache);
   acc.xyz += ptSample;
   acc.x += projTri.forwardPdf(u2, projTriCache);
   acc.x += projTri.forwardWeight(u2, projTriCache);
   acc.x += projTri.backwardPdf(ptSample);
   acc.x += projTri.backwardWeight(ptSample);

   output[0] = acc;
}
