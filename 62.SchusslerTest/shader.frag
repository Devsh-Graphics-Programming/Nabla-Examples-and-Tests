#version 430 core
// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


layout (location = 0) in vec3 Normal;
layout (location = 1) in vec3 Pos;
layout (location = 2) flat in vec3 LightPos;
layout (location = 3) flat in vec3 CamPos;
layout (location = 4) flat in vec3 GNormal;
layout (location = 5) flat in float Alpha;
layout (location = 6) flat in float AlphaY;

layout (location = 0) out vec4 outColor;

layout (push_constant) uniform PC {
    layout (offset = 64) vec3 campos;
    layout (offset = 80) uint testNum;
} pc;

#define TEST_GGX 1
#define TEST_BECKMANN 2
#define TEST_PHONG 3
#define TEST_AS 4
#define TEST_OREN_NAYAR 5
#define TEST_LAMBERT 6

#include <nbl/builtin/glsl/bxdf/brdf/specular/ggx.glsl>
#include <nbl/builtin/glsl/bxdf/brdf/specular/beckmann.glsl>
#include <nbl/builtin/glsl/bxdf/brdf/specular/blinn_phong.glsl>
#include <nbl/builtin/glsl/bxdf/brdf/diffuse/oren_nayar.glsl>
#include <nbl/builtin/glsl/bxdf/brdf/diffuse/lambert.glsl>

float pdot(vec3 a, vec3 b) {
    return max(0.0, dot(a,b));
}

float ap(vec3 wi, vec3 wp, vec3 wg) {
    return pdot(wi, wp)/pdot(wp, wg);
}

float at(vec3 wi, vec3 wt, vec3 wp, vec3 wg) {
    return pdot(wi, wt) * sqrt(1-(pdot(wp,wg)*pdot(wp,wg))) / pdot(wp,wg);
}

float lambdap(vec3 wi, vec3 wt, vec3 wp, vec3 wg) {
    return ap(wi, wp, wg) / (ap(wi,wp,wg) + at(wi,wt,wp,wg));
}

float lambdat(vec3 wi, vec3 wt, vec3 wp, vec3 wg) {
    return at(wi,wt,wp,wg) / (ap(wi,wp,wg) + at(wi,wt,wp,wg));
}

float G1(vec3 wo, vec3 wp, float a2) {
    float dotNV = pdot(wp,wo);
    float denomC = sqrt(a2 + (1.0f - a2) * dotNV * dotNV) + dotNV;

    return 2.0f * dotNV / denomC;
}

vec3 evalBRDF(vec3 wi, vec3 wo, vec3 N) {
    const mat2x3 ior = mat2x3(vec3(1.02,1.3,1.02), vec3(1.0,2.0,1.0));
    const vec3 albedo = vec3(0.5);
    float a2 = Alpha;
    nbl_glsl_IsotropicViewSurfaceInteraction inter_ = nbl_glsl_calcSurfaceInteractionFromViewVector(wi, N );
    nbl_glsl_AnisotropicViewSurfaceInteraction inter = nbl_glsl_calcAnisotropicInteraction(inter_);
    nbl_glsl_LightSample _sample = nbl_glsl_createLightSample(wo,inter);
    nbl_glsl_AnisotropicMicrofacetCache cache = nbl_glsl_calcAnisotropicMicrofacetCache(inter, _sample);
    vec3 brdf = vec3(0.0);
    if (pc.testNum == TEST_GGX) {
        brdf = nbl_glsl_ggx_height_correlated_cos_eval(_sample, inter_, cache.isotropic, ior, a2);
    } else if (pc.testNum == TEST_BECKMANN) {
        brdf = nbl_glsl_beckmann_height_correlated_cos_eval(_sample, inter_, cache.isotropic, ior, a2);
    } else if (pc.testNum == TEST_PHONG) {
        float n = nbl_glsl_alpha2_to_phong_exp(a2);
        brdf = nbl_glsl_blinn_phong_cos_eval(_sample, inter_, cache.isotropic, n, ior);
    } else if (pc.testNum == TEST_AS) {
        float nx = nbl_glsl_alpha2_to_phong_exp(a2);
        float aa = 1.0-Alpha;
        float ny = nbl_glsl_alpha2_to_phong_exp(aa*aa);
        brdf = nbl_glsl_blinn_phong_cos_eval(_sample, inter, cache, nx, ny, ior);
    } else if (pc.testNum == TEST_OREN_NAYAR) {
        brdf = albedo*nbl_glsl_oren_nayar_cos_eval(_sample, inter_, a2);
    } else if (pc.testNum == TEST_LAMBERT) {
        brdf = albedo*nbl_glsl_lambertian_cos_eval(_sample);
    }
    return brdf;
}

void main()
{
    const float Intensity = 20.0;

    vec3 L = LightPos-Pos;
    vec3 Lnorm = normalize(L);
    const float THRESHOLD = AlphaY+0.5;
    float a2 = Alpha;
    float a2formeta = AlphaY;
    
    vec3 wi = normalize(CamPos - Pos);
    vec3 wo = Lnorm;
    const float deviation = dot(Normal,GNormal);
    vec3 wt = -normalize(Normal-deviation*GNormal);
    vec3 wdasho = normalize(reflect(-wo, wt));
    vec3 wdashi = normalize(reflect(-wi, wt));

    const vec3 albedo = vec3(0.5);
    vec3 brdf_cos_wi_wo = evalBRDF(wi,wo,Normal);
    vec3 brdf_cos_wi_wdasho = evalBRDF(wi,wdasho,Normal);
    vec3 brdf_cos_wdashi_wo = evalBRDF(wdashi,wo,Normal);
    vec3 brdf;
    if (deviation<THRESHOLD) 
    {
        brdf = lambdap(wi, wt, Normal, GNormal) * (
            G1(wo, Normal, a2formeta) * brdf_cos_wi_wo +
            G1(wo, wt, a2formeta) * (1.0 - G1(wdasho, Normal, a2formeta)) * brdf_cos_wi_wdasho
        ) + lambdat(wi, wt, Normal, GNormal) * brdf_cos_wdashi_wo * G1(wo, Normal, a2formeta);
    } else {
        brdf = brdf_cos_wi_wo;
    }
    const vec3 col = Intensity*brdf/dot(L,L);
    //red output means brdf>1.0
    //outColor = any(greaterThan(brdf,vec3(1.0))) ? vec4(1.0,0.0,0.0,1.0) : vec4(Intensity*brdf/dot(L,L), 1.0);
    outColor = vec4(col, 1.0);
    //outColor = (inter_.NdotV<0.0||_sample.NdotL<0.0) ? vec4(1.0,0.0,0.0,1.0) : vec4(col, 1.0);
}