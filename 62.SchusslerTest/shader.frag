#version 430 core
// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


layout (location = 0) in vec3 Normal;
layout (location = 1) in vec3 Pos;
layout (location = 2) flat in vec3 LightPos;
layout (location = 3) flat in vec3 CamPos;
layout (location = 4) flat in float Alpha;

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
    return pdot(wi, wt) * sqrt(1.0-(pdot(wp,wg)*pdot(wp,wg))) / pdot(wp,wg);
}

float lambdap(vec3 wi, vec3 wt, vec3 wp, vec3 wg) {
    return ap(wi, wp, wg) / (ap(wi,wp,wg) + at(wi,wt,wp,wg));
}

float lambdat(vec3 wi, vec3 wt, vec3 wp, vec3 wg) {
    return at(wi,wt,wp,wg) / (ap(wi,wp,wg) + at(wi,wt,wp,wg));
}

float G1(vec3 wi, vec3 wm, vec3 wt, vec3 wp, vec3 wg) {
    if (dot(wi,wm) > 0.f) {
        return min(1.f, pdot(wi, wg)/(lambdap(wi,wt,wp,wg) + lambdat(wi,wt,wp,wg)));
    }
    return 0.f;
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
    const float THRESHOLD = 1.0 - 1e-5;

    vec3 dFdxPos = dFdx(Pos);
    vec3 dFdyPos = dFdy(Pos);
    vec3 GN = normalize(cross(dFdxPos,dFdyPos));
    vec3 N = normalize(Normal);
    vec3 L = LightPos-Pos;

    vec3 wi = normalize(CamPos - Pos);
    vec3 wo = normalize(L);
    const float deviation = dot(N,GN);
    if (deviation < 0) {
        outColor = vec4(1.0, 0.0, 0.0, 1.0);
        return;
    } 
    vec3 wt = -normalize(N-deviation*GN);
    vec3 wdasho = reflect(wo, wt);
    vec3 wdashi = reflect(wi, wt);

    float a2 = Alpha;
    const float NdotV = dot(GN,wi);
    const float NdotL = dot(GN,wo);
    if (NdotV<nbl_glsl_FLT_MIN || NdotL<nbl_glsl_FLT_MIN) {
        outColor = vec4(0.0,0.0,0.0,1.0);
        return;
    }

    vec3 brdf_cos_wi_wo = evalBRDF(wi,wo,N);
    vec3 brdf_cos_wi_wdasho = evalBRDF(wi,wdasho,N);
    vec3 brdf_cos_wdashi_wo = evalBRDF(wdashi,wo,N);
    vec3 brdf = vec3(0.0);

    if (deviation < THRESHOLD) 
    {
        if (dot(wi, GN) >= 0.0) brdf += lambdap(wi, wt, N, GN) * G1(wo, N, wt, N, GN) * brdf_cos_wi_wo;
        if (dot(wi, GN) >= 0.0) brdf += lambdap(wi, wt, N, GN) * G1(wo, wt,  wt, N, GN) 
            * (1.0 - G1(wdasho, N,  wt, N, GN)) * brdf_cos_wi_wdasho;
        if (dot(wdashi, GN) >= 0.0) brdf += lambdat(wi, wt, N, GN) * brdf_cos_wdashi_wo * G1(wo, N,  wt, N, GN);
    } else {
        brdf = brdf_cos_wi_wo;
    }
    const vec3 col = Intensity*brdf/dot(L,L);
    // if (AlphaY >= 0.8) 
    //     outColor = vec4(col, 1.0);
    // else if (AlphaY >= 0.6) 
    //     outColor = vec4(wdashi, 1.0);
    // else if (AlphaY >= 0.4)
    //     outColor = vec4(-wi, 1.0);
    // else if (AlphaY >= 0.2)
    //     outColor = vec4(wt, 1.0);
    // else
    outColor = vec4(col, 1.0);
}