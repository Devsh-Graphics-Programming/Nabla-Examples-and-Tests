#ifndef _NBL_HLSL_EXT_NEXT_EVENT_ESTIMATOR_INCLUDED_
#define _NBL_HLSL_EXT_NEXT_EVENT_ESTIMATOR_INCLUDED_

#include "common.hlsl"

namespace nbl
{
namespace hlsl
{
namespace ext
{
namespace NextEventEstimator
{

template<ProceduralShapeType PST, PTPolygonMethod PPM>
struct ShapeSampling;

template<PTPolygonMethod PPM>
struct ShapeSampling<PST_SPHERE, PPM>
{
    static ShapeSampling<PST_SPHERE, PPM> create(NBL_CONST_REF_ARG(Shape<PST_SPHERE>) sphere)
    {
        ShapeSampling<PST_SPHERE, PPM> retval;
        retval.sphere = sphere;
        return retval;
    }

    template<typename Ray>
    float deferredPdf(NBL_CONST_REF_ARG(Ray) ray)
    {
        return 1.0 / sphere.getSolidAngle(ray.origin);
    }

    template<class Aniso>
    float32_t3 generate_and_pdf(NBL_REF_ARG(float32_t) pdf, NBL_REF_ARG(float32_t) newRayMaxT, NBL_CONST_REF_ARG(float32_t3) origin, NBL_CONST_REF_ARG(Aniso) interaction, bool isBSDF, NBL_CONST_REF_ARG(float32_t3) xi)
    {
        float32_t3 Z = sphere.position - origin;
        const float distanceSQ = hlsl::dot<float32_t3>(Z,Z);
        const float cosThetaMax2 = 1.0 - sphere.radius2 / distanceSQ;
        if (cosThetaMax2 > 0.0)
        {
            const float rcpDistance = 1.0 / hlsl::sqrt<float32_t>(distanceSQ);
            Z *= rcpDistance;

            const float cosThetaMax = hlsl::sqrt<float32_t>(cosThetaMax2);
            const float cosTheta = hlsl::mix(1.0f, cosThetaMax, xi.x);

            float32_t3 L = Z * cosTheta;

            const float cosTheta2 = cosTheta * cosTheta;
            const float sinTheta = hlsl::sqrt<float32_t>(1.0 - cosTheta2);
            float sinPhi, cosPhi;
            math::sincos<float>(2.0 * numbers::pi<float> * xi.y - numbers::pi<float>, sinPhi, cosPhi);
            float32_t3 X, Y;
            math::frisvad<float32_t3>(Z, X, Y);

            L += (X * cosPhi + Y * sinPhi) * sinTheta;

            newRayMaxT = (cosTheta - hlsl::sqrt<float32_t>(cosTheta2 - cosThetaMax2)) / rcpDistance;
            pdf = 1.0 / (2.0 * numbers::pi<float> * (1.0 - cosThetaMax));
            return L;
        }
        pdf = 0.0;
        return float32_t3(0.0,0.0,0.0);
    }

    Shape<PST_SPHERE> sphere;
};

template<>
struct ShapeSampling<PST_TRIANGLE, PPM_AREA>
{
    static ShapeSampling<PST_TRIANGLE, PPM_AREA> create(NBL_CONST_REF_ARG(Shape<PST_TRIANGLE>) tri)
    {
        ShapeSampling<PST_TRIANGLE, PPM_AREA> retval;
        retval.tri = tri;
        return retval;
    }

    template<typename Ray>
    float deferredPdf(NBL_CONST_REF_ARG(Ray) ray)
    {
        const float dist = ray.intersectionT;
        const float32_t3 L = ray.direction;
        return dist * dist / hlsl::abs<float32_t>(hlsl::dot<float32_t3>(tri.getNormalTimesArea(), L));
    }

    template<class Aniso>
    float32_t3 generate_and_pdf(NBL_REF_ARG(float32_t) pdf, NBL_REF_ARG(float32_t) newRayMaxT, NBL_CONST_REF_ARG(float32_t3) origin, NBL_CONST_REF_ARG(Aniso) interaction, bool isBSDF, NBL_CONST_REF_ARG(float32_t3) xi)
    {
        const float32_t3 edge0 = tri.vertex1 - tri.vertex0;
        const float32_t3 edge1 = tri.vertex2 - tri.vertex0;
        const float sqrtU = hlsl::sqrt<float32_t>(xi.x);
        float32_t3 pnt = tri.vertex0 + edge0 * (1.0 - sqrtU) + edge1 * sqrtU * xi.y;
        float32_t3 L = pnt - origin;

        const float distanceSq = hlsl::dot<float32_t3>(L,L);
        const float rcpDistance = 1.0 / hlsl::sqrt<float32_t>(distanceSq);
        L *= rcpDistance;

        pdf = distanceSq / hlsl::abs<float32_t>(hlsl::dot<float32_t3>(hlsl::cross<float32_t3>(edge0, edge1) * 0.5f, L));
        newRayMaxT = 1.0 / rcpDistance;
        return L;
    }

    Shape<PST_TRIANGLE> tri;
};

template<>
struct ShapeSampling<PST_TRIANGLE, PPM_SOLID_ANGLE>
{
    static ShapeSampling<PST_TRIANGLE, PPM_SOLID_ANGLE> create(NBL_CONST_REF_ARG(Shape<PST_TRIANGLE>) tri)
    {
        ShapeSampling<PST_TRIANGLE, PPM_SOLID_ANGLE> retval;
        retval.tri = tri;
        return retval;
    }

    template<typename Ray>
    float deferredPdf(NBL_CONST_REF_ARG(Ray) ray)
    {
        shapes::SphericalTriangle<float> st = shapes::SphericalTriangle<float>::create(tri.vertex0, tri.vertex1, tri.vertex2, ray.origin);
        const float rcpProb = st.solidAngleOfTriangle();
        // if `rcpProb` is NAN then the triangle's solid angle was close to 0.0
        return rcpProb > numeric_limits<float>::min ? (1.0 / rcpProb) : numeric_limits<float>::max;
    }

    template<class Aniso>
    float32_t3 generate_and_pdf(NBL_REF_ARG(float32_t) pdf, NBL_REF_ARG(float32_t) newRayMaxT, NBL_CONST_REF_ARG(float32_t3) origin, NBL_CONST_REF_ARG(Aniso) interaction, bool isBSDF, NBL_CONST_REF_ARG(float32_t3) xi)
    {
        float rcpPdf;
        shapes::SphericalTriangle<float> st = shapes::SphericalTriangle<float>::create(tri.vertex0, tri.vertex1, tri.vertex2, origin);
        sampling::SphericalTriangle<float> sst = sampling::SphericalTriangle<float>::create(st);

        const float32_t3 L = sst.generate(rcpPdf, xi.xy);

        pdf = rcpPdf > numeric_limits<float>::min ? (1.0 / rcpPdf) : numeric_limits<float>::max;

        const float32_t3 N = tri.getNormalTimesArea();
        newRayMaxT = hlsl::dot<float32_t3>(N, tri.vertex0 - origin) / hlsl::dot<float32_t3>(N, L);
        return L;
    }

    Shape<PST_TRIANGLE> tri;
};

template<>
struct ShapeSampling<PST_TRIANGLE, PPM_APPROX_PROJECTED_SOLID_ANGLE>
{
    static ShapeSampling<PST_TRIANGLE, PPM_APPROX_PROJECTED_SOLID_ANGLE> create(NBL_CONST_REF_ARG(Shape<PST_TRIANGLE>) tri)
    {
        ShapeSampling<PST_TRIANGLE, PPM_APPROX_PROJECTED_SOLID_ANGLE> retval;
        retval.tri = tri;
        return retval;
    }

    template<typename Ray>
    float deferredPdf(NBL_CONST_REF_ARG(Ray) ray)
    {
        const float32_t3 L = ray.direction;
        shapes::SphericalTriangle<float> st = shapes::SphericalTriangle<float>::create(tri.vertex0, tri.vertex1, tri.vertex2, ray.origin);
        sampling::ProjectedSphericalTriangle<float> pst = sampling::ProjectedSphericalTriangle<float>::create(st);
        const float pdf = pst.pdf(ray.normalAtOrigin, ray.wasBSDFAtOrigin, L);
        // if `pdf` is NAN then the triangle's projected solid angle was close to 0.0, if its close to INF then the triangle was very small
        return pdf < numeric_limits<float>::max ? pdf : numeric_limits<float>::max;
    }

    template<class Aniso>
    float32_t3 generate_and_pdf(NBL_REF_ARG(float32_t) pdf, NBL_REF_ARG(float32_t) newRayMaxT, NBL_CONST_REF_ARG(float32_t3) origin, NBL_CONST_REF_ARG(Aniso) interaction, bool isBSDF, NBL_CONST_REF_ARG(float32_t3) xi)
    {
        float rcpPdf;
        shapes::SphericalTriangle<float> st = shapes::SphericalTriangle<float>::create(tri.vertex0, tri.vertex1, tri.vertex2, origin);
        sampling::ProjectedSphericalTriangle<float> sst = sampling::ProjectedSphericalTriangle<float>::create(st);

        const float32_t3 L = sst.generate(rcpPdf, interaction.isotropic.N, isBSDF, xi.xy);

        pdf = rcpPdf > numeric_limits<float>::min ? (1.0 / rcpPdf) : numeric_limits<float>::max;

        const float32_t3 N = tri.getNormalTimesArea();
        newRayMaxT = hlsl::dot<float32_t3>(N, tri.vertex0 - origin) / hlsl::dot<float32_t3>(N, L);
        return L;
    }

    Shape<PST_TRIANGLE> tri;
};

template<>
struct ShapeSampling<PST_RECTANGLE, PPM_AREA>
{
    static ShapeSampling<PST_RECTANGLE, PPM_AREA> create(NBL_CONST_REF_ARG(Shape<PST_RECTANGLE>) rect)
    {
        ShapeSampling<PST_RECTANGLE, PPM_AREA> retval;
        retval.rect = rect;
        return retval;
    }

    template<typename Ray>
    float deferredPdf(NBL_CONST_REF_ARG(Ray) ray)
    {
        const float dist = ray.intersectionT;
        const float32_t3 L = ray.direction;
        return dist * dist / hlsl::abs<float32_t>(hlsl::dot<float32_t3>(rect.getNormalTimesArea(), L));
    }

    template<class Aniso>
    float32_t3 generate_and_pdf(NBL_REF_ARG(float32_t) pdf, NBL_REF_ARG(float32_t) newRayMaxT, NBL_CONST_REF_ARG(float32_t3) origin, NBL_CONST_REF_ARG(Aniso) interaction, bool isBSDF, NBL_CONST_REF_ARG(float32_t3) xi)
    {
        const float32_t3 N = rect.getNormalTimesArea();
        const float32_t3 origin2origin = rect.offset - origin;

        float32_t3 L = origin2origin + rect.edge0 * xi.x + rect.edge1 * xi.y;
        const float distSq = hlsl::dot<float32_t3>(L, L);
        const float rcpDist = 1.0 / hlsl::sqrt<float32_t>(distSq);
        L *= rcpDist;
        pdf = distSq / hlsl::abs<float32_t>(hlsl::dot<float32_t3>(N, L));
        newRayMaxT = 1.0 / rcpDist;
        return L;
    }

    Shape<PST_RECTANGLE> rect;
};

template<>
struct ShapeSampling<PST_RECTANGLE, PPM_SOLID_ANGLE>
{
    static ShapeSampling<PST_RECTANGLE, PPM_SOLID_ANGLE> create(NBL_CONST_REF_ARG(Shape<PST_RECTANGLE>) rect)
    {
        ShapeSampling<PST_RECTANGLE, PPM_SOLID_ANGLE> retval;
        retval.rect = rect;
        return retval;
    }

    template<typename Ray>
    float deferredPdf(NBL_CONST_REF_ARG(Ray) ray)
    {
        float pdf;
        float32_t3x3 rectNormalBasis;
        float32_t2 rectExtents;
        rect.getNormalBasis(rectNormalBasis, rectExtents);
        shapes::SphericalRectangle<float> sphR0 = shapes::SphericalRectangle<float>::create(ray.origin, rect.offset, rectNormalBasis);
        float solidAngle = sphR0.solidAngleOfRectangle(rectExtents);
        if (solidAngle > numeric_limits<float>::min)
            pdf = 1.f / solidAngle;
        else
            pdf = bit_cast<float>(numeric_limits<float>::infinity);
        return pdf;
    }

    template<class Aniso>
    float32_t3 generate_and_pdf(NBL_REF_ARG(float32_t) pdf, NBL_REF_ARG(float32_t) newRayMaxT, NBL_CONST_REF_ARG(float32_t3) origin, NBL_CONST_REF_ARG(Aniso) interaction, bool isBSDF, NBL_CONST_REF_ARG(float32_t3) xi)
    {
        const float32_t3 N = rect.getNormalTimesArea();
        const float32_t3 origin2origin = rect.offset - origin;

        float32_t3x3 rectNormalBasis;
        float32_t2 rectExtents;
        rect.getNormalBasis(rectNormalBasis, rectExtents);
        shapes::SphericalRectangle<float> sphR0 = shapes::SphericalRectangle<float>::create(origin, rect.offset, rectNormalBasis);
        float32_t3 L = hlsl::promote<float32_t3>(0.0);
        float solidAngle = sphR0.solidAngleOfRectangle(rectExtents);

        sampling::SphericalRectangle<float> ssph = sampling::SphericalRectangle<float>::create(sphR0);
        float32_t2 sphUv = ssph.generate(rectExtents, xi.xy, solidAngle);
        if (solidAngle > numeric_limits<float>::min)
        {
            float32_t3 sph_sample = sphUv.x * rect.edge0 + sphUv.y * rect.edge1 + rect.offset;
            L = sph_sample - origin;
            const bool invalid = hlsl::all(hlsl::abs(L) < hlsl::promote<float32_t3>(numeric_limits<float>::min));
            L = hlsl::mix(hlsl::normalize(L), hlsl::promote<float32_t3>(0.0), invalid);
            pdf = hlsl::mix(1.f / solidAngle, bit_cast<float>(numeric_limits<float>::infinity), invalid);
        }
        else
            pdf = bit_cast<float>(numeric_limits<float>::infinity);

        newRayMaxT = hlsl::dot<float32_t3>(N, origin2origin) / hlsl::dot<float32_t3>(N, L);
        return L;
    }

    Shape<PST_RECTANGLE> rect;
};

// PPM_APPROX_PROJECTED_SOLID_ANGLE not available for PST_TRIANGLE


template<class Scene, typename Ray, class LightSample, class Aniso, IntersectMode Mode, ProceduralShapeType PST, PTPolygonMethod PPM>
struct Estimator;

template<class Scene, typename Ray, class LightSample, class Aniso, PTPolygonMethod PPM>
struct Estimator<Scene, Ray, LightSample, Aniso, IM_PROCEDURAL, PST_SPHERE, PPM>
{
    using scalar_type = typename Ray::scalar_type;
    using vector3_type = vector<scalar_type, 3>;
    using ray_type = Ray;
    using scene_type = Scene;
    using light_type = typename Scene::light_type;
    using spectral_type = typename light_type::spectral_type;
    using interaction_type = Aniso;
    using quotient_pdf_type = sampling::quotient_and_pdf<spectral_type, scalar_type>;
    using sample_type = LightSample;
    using ray_dir_info_type = typename sample_type::ray_dir_info_type;

    // affected by https://github.com/microsoft/DirectXShaderCompiler/issues/7007
    // NBL_CONSTEXPR_STATIC_INLINE PTPolygonMethod PolygonMethod = PPM;
    enum : uint16_t { PolygonMethod = PPM };

    static spectral_type deferredEvalAndPdf(NBL_REF_ARG(scalar_type) pdf, NBL_CONST_REF_ARG(scene_type) scene, uint32_t lightID, NBL_CONST_REF_ARG(ray_type) ray)
    {
        pdf = 1.0 / scene.lightCount;
        const light_type light = scene.lights[lightID];
        const Shape<PST_SPHERE> sphere = scene.spheres[light.objectID.id];
        const ShapeSampling<PST_SPHERE, PPM> sampling = ShapeSampling<PST_SPHERE, PPM>::create(sphere);
        pdf *= sampling.template deferredPdf<ray_type>(ray);

        return light.radiance;
    }

    static sample_type generate_and_quotient_and_pdf(NBL_REF_ARG(quotient_pdf_type) quotient_pdf, NBL_REF_ARG(scalar_type) newRayMaxT, NBL_CONST_REF_ARG(scene_type) scene, uint32_t lightID, NBL_CONST_REF_ARG(vector3_type) origin, NBL_CONST_REF_ARG(interaction_type) interaction, bool isBSDF, NBL_CONST_REF_ARG(vector3_type) xi, uint32_t depth)
    {
        const light_type light = scene.lights[lightID];
        const Shape<PST_SPHERE> sphere = scene.spheres[light.objectID.id];
        const ShapeSampling<PST_SPHERE, PPM> sampling = ShapeSampling<PST_SPHERE, PPM>::create(sphere);

        scalar_type pdf;
        const vector3_type sampleL = sampling.template generate_and_pdf<interaction_type>(pdf, newRayMaxT, origin, interaction, isBSDF, xi);
        const vector3_type N = interaction.getN();
        const scalar_type NdotL = nbl::hlsl::dot<vector3_type>(N, sampleL);
        ray_dir_info_type rayL;
        rayL.setDirection(sampleL);
        sample_type L = sample_type::create(rayL,interaction.getT(),interaction.getB(),NdotL);

        newRayMaxT *= Tolerance<scalar_type>::getEnd(depth);
        pdf *= 1.0 / scalar_type(scene.lightCount);
        spectral_type quo = light.radiance / pdf;
        quotient_pdf = quotient_pdf_type::create(quo, pdf);

        return L;
    }
};

template<class Scene, typename Ray, class LightSample, class Aniso, PTPolygonMethod PPM>
struct Estimator<Scene, Ray, LightSample, Aniso, IM_PROCEDURAL, PST_TRIANGLE, PPM>
{
    using scalar_type = typename Ray::scalar_type;
    using vector3_type = vector<scalar_type, 3>;
    using ray_type = Ray;
    using scene_type = Scene;
    using light_type = typename Scene::light_type;
    using spectral_type = typename light_type::spectral_type;
    using interaction_type = Aniso;
    using quotient_pdf_type = sampling::quotient_and_pdf<spectral_type, scalar_type>;
    using sample_type = LightSample;
    using ray_dir_info_type = typename sample_type::ray_dir_info_type;

    // NBL_CONSTEXPR_STATIC_INLINE PTPolygonMethod PolygonMethod = PPM;
    enum : uint16_t { PolygonMethod = PPM };

    static spectral_type deferredEvalAndPdf(NBL_REF_ARG(scalar_type) pdf, NBL_CONST_REF_ARG(scene_type) scene, uint32_t lightID, NBL_CONST_REF_ARG(ray_type) ray)
    {
        pdf = 1.0 / scene.lightCount;
        const light_type light = scene.lights[lightID];
        const Shape<PST_TRIANGLE> tri = scene.triangles[light.objectID.id];
        const ShapeSampling<PST_TRIANGLE, PPM> sampling = ShapeSampling<PST_TRIANGLE, PPM>::create(tri);
        pdf *= sampling.template deferredPdf<ray_type>(ray);

        return light.radiance;
    }

    static sample_type generate_and_quotient_and_pdf(NBL_REF_ARG(quotient_pdf_type) quotient_pdf, NBL_REF_ARG(scalar_type) newRayMaxT, NBL_CONST_REF_ARG(scene_type) scene, uint32_t lightID, NBL_CONST_REF_ARG(vector3_type) origin, NBL_CONST_REF_ARG(interaction_type) interaction, bool isBSDF, NBL_CONST_REF_ARG(vector3_type) xi, uint32_t depth)
    {
        const light_type light = scene.lights[lightID];
        const Shape<PST_TRIANGLE> tri = scene.triangles[light.objectID.id];
        const ShapeSampling<PST_TRIANGLE, PPM> sampling = ShapeSampling<PST_TRIANGLE, PPM>::create(tri);

        scalar_type pdf;
        const vector3_type sampleL = sampling.template generate_and_pdf<interaction_type>(pdf, newRayMaxT, origin, interaction, isBSDF, xi);
        const vector3_type N = interaction.getN();
        const scalar_type NdotL = nbl::hlsl::dot<vector3_type>(N, sampleL);
        ray_dir_info_type rayL;
        rayL.setDirection(sampleL);
        sample_type L = sample_type::create(rayL,interaction.getT(),interaction.getB(),NdotL);

        newRayMaxT *= Tolerance<scalar_type>::getEnd(depth);
        pdf *= 1.0 / scalar_type(scene.lightCount);
        spectral_type quo = light.radiance / pdf;
        quotient_pdf = quotient_pdf_type::create(quo, pdf);

        return L;
    }
};

template<typename Scene, typename Ray, class LightSample, class Aniso, PTPolygonMethod PPM>
struct Estimator<Scene, Ray, LightSample, Aniso, IM_PROCEDURAL, PST_RECTANGLE, PPM>
{
    using scalar_type = typename Ray::scalar_type;
    using vector3_type = vector<scalar_type, 3>;
    using ray_type = Ray;
    using scene_type = Scene;
    using light_type = typename Scene::light_type;
    using spectral_type = typename light_type::spectral_type;
    using interaction_type = Aniso;
    using quotient_pdf_type = sampling::quotient_and_pdf<spectral_type, scalar_type>;
    using sample_type = LightSample;
    using ray_dir_info_type = typename sample_type::ray_dir_info_type;

    // NBL_CONSTEXPR_STATIC_INLINE PTPolygonMethod PolygonMethod = PPM;
    enum : uint16_t { PolygonMethod = PPM };

    static spectral_type deferredEvalAndPdf(NBL_REF_ARG(scalar_type) pdf, NBL_CONST_REF_ARG(scene_type) scene, uint32_t lightID, NBL_CONST_REF_ARG(ray_type) ray)
    {
        pdf = 1.0 / scene.lightCount;
        const light_type light = scene.lights[lightID];
        const Shape<PST_RECTANGLE> rect = scene.rectangles[light.objectID.id];
        const ShapeSampling<PST_RECTANGLE, PPM> sampling = ShapeSampling<PST_RECTANGLE, PPM>::create(rect);
        pdf *= sampling.template deferredPdf<ray_type>(ray);

        return light.radiance;
    }

    static sample_type generate_and_quotient_and_pdf(NBL_REF_ARG(quotient_pdf_type) quotient_pdf, NBL_REF_ARG(scalar_type) newRayMaxT, NBL_CONST_REF_ARG(scene_type) scene, uint32_t lightID, NBL_CONST_REF_ARG(vector3_type) origin, NBL_CONST_REF_ARG(interaction_type) interaction, bool isBSDF, NBL_CONST_REF_ARG(vector3_type) xi, uint32_t depth)
    {
        const light_type light = scene.lights[lightID];
        const Shape<PST_RECTANGLE> rect = scene.rectangles[light.objectID.id];
        const ShapeSampling<PST_RECTANGLE, PPM> sampling = ShapeSampling<PST_RECTANGLE, PPM>::create(rect);

        scalar_type pdf;
        const vector3_type sampleL = sampling.template generate_and_pdf<interaction_type>(pdf, newRayMaxT, origin, interaction, isBSDF, xi);
        ray_dir_info_type rayL;
        if (hlsl::isinf(pdf))
        {
            quotient_pdf = quotient_pdf_type::create(hlsl::promote<spectral_type>(0.0), 0.0);
            return sample_type::createInvalid();
        }

        const vector3_type N = interaction.getN();
        const scalar_type NdotL = nbl::hlsl::dot<vector3_type>(N, sampleL);
        
        rayL.setDirection(sampleL);
        sample_type L = sample_type::create(rayL,interaction.getT(),interaction.getB(),NdotL);

        newRayMaxT *= Tolerance<scalar_type>::getEnd(depth);
        pdf *= 1.0 / scalar_type(scene.lightCount);
        spectral_type quo = light.radiance / pdf;
        quotient_pdf = quotient_pdf_type::create(quo, pdf);

        return L;
    }
};

}
}
}
}

#endif
