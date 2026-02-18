#ifndef _PATHTRACER_EXAMPLE_NEXT_EVENT_ESTIMATOR_INCLUDED_
#define _PATHTRACER_EXAMPLE_NEXT_EVENT_ESTIMATOR_INCLUDED_

#include "example_common.hlsl"

using namespace nbl;
using namespace hlsl;

template<typename T, ProceduralShapeType PST, PTPolygonMethod PPM>
struct ShapeSampling;

template<typename T, PTPolygonMethod PPM>
struct ShapeSampling<T, PST_SPHERE, PPM>
{
    using scalar_type = T;
    using vector3_type = vector<T, 3>;

    static ShapeSampling<T, PST_SPHERE, PPM> create(NBL_CONST_REF_ARG(Shape<T, PST_SPHERE>) sphere)
    {
        ShapeSampling<T, PST_SPHERE, PPM> retval;
        retval.sphere = sphere;
        return retval;
    }

    template<typename Ray>
    scalar_type deferredPdf(NBL_CONST_REF_ARG(Ray) ray)
    {
        return 1.0 / sphere.getSolidAngle(ray.origin);
    }

    template<class Aniso>
    vector3_type generate_and_pdf(NBL_REF_ARG(scalar_type) pdf, NBL_REF_ARG(scalar_type) newRayMaxT, NBL_CONST_REF_ARG(vector3_type) origin, NBL_CONST_REF_ARG(Aniso) interaction, bool isBSDF, NBL_CONST_REF_ARG(vector3_type) xi)
    {
        vector3_type Z = sphere.position - origin;
        const scalar_type distanceSQ = hlsl::dot<vector3_type>(Z,Z);
        const scalar_type cosThetaMax2 = 1.0 - sphere.radius2 / distanceSQ;
        if (cosThetaMax2 > 0.0)
        {
            const scalar_type rcpDistance = 1.0 / hlsl::sqrt<scalar_type>(distanceSQ);
            Z *= rcpDistance;

            const scalar_type cosThetaMax = hlsl::sqrt<scalar_type>(cosThetaMax2);
            const scalar_type cosTheta = hlsl::mix(1.0f, cosThetaMax, xi.x);

            vector3_type L = Z * cosTheta;

            const scalar_type cosTheta2 = cosTheta * cosTheta;
            const scalar_type sinTheta = hlsl::sqrt<scalar_type>(1.0 - cosTheta2);
            scalar_type sinPhi, cosPhi;
            math::sincos<scalar_type>(2.0 * numbers::pi<scalar_type> * xi.y - numbers::pi<scalar_type>, sinPhi, cosPhi);
            vector3_type X, Y;
            math::frisvad<vector3_type>(Z, X, Y);

            L += (X * cosPhi + Y * sinPhi) * sinTheta;

            newRayMaxT = (cosTheta - hlsl::sqrt<scalar_type>(cosTheta2 - cosThetaMax2)) / rcpDistance;
            pdf = 1.0 / (2.0 * numbers::pi<scalar_type> * (1.0 - cosThetaMax));
            return L;
        }
        pdf = 0.0;
        return vector3_type(0.0,0.0,0.0);
    }

    Shape<T, PST_SPHERE> sphere;
};

template<typename T>
struct ShapeSampling<T, PST_TRIANGLE, PPM_AREA>
{
    using scalar_type = T;
    using vector3_type = vector<T, 3>;

    static ShapeSampling<T, PST_TRIANGLE, PPM_AREA> create(NBL_CONST_REF_ARG(Shape<T, PST_TRIANGLE>) tri)
    {
        ShapeSampling<T, PST_TRIANGLE, PPM_AREA> retval;
        retval.tri = tri;
        return retval;
    }

    template<typename Ray>
    scalar_type deferredPdf(NBL_CONST_REF_ARG(Ray) ray)
    {
        const scalar_type dist = ray.intersectionT;
        const vector3_type L = ray.direction;
        return dist * dist / hlsl::abs<scalar_type>(hlsl::dot<vector3_type>(tri.getNormalTimesArea(), L));
    }

    template<class Aniso>
    vector3_type generate_and_pdf(NBL_REF_ARG(scalar_type) pdf, NBL_REF_ARG(scalar_type) newRayMaxT, NBL_CONST_REF_ARG(vector3_type) origin, NBL_CONST_REF_ARG(Aniso) interaction, bool isBSDF, NBL_CONST_REF_ARG(vector3_type) xi)
    {
        const vector3_type edge0 = tri.vertex1 - tri.vertex0;
        const vector3_type edge1 = tri.vertex2 - tri.vertex0;
        const scalar_type sqrtU = hlsl::sqrt<scalar_type>(xi.x);
        vector3_type pnt = tri.vertex0 + edge0 * (1.0 - sqrtU) + edge1 * sqrtU * xi.y;
        vector3_type L = pnt - origin;

        const scalar_type distanceSq = hlsl::dot<vector3_type>(L,L);
        const scalar_type rcpDistance = 1.0 / hlsl::sqrt<scalar_type>(distanceSq);
        L *= rcpDistance;

        pdf = distanceSq / hlsl::abs<scalar_type>(hlsl::dot<vector3_type>(hlsl::cross<vector3_type>(edge0, edge1) * 0.5f, L));
        newRayMaxT = 1.0 / rcpDistance;
        return L;
    }

    Shape<T, PST_TRIANGLE> tri;
};

template<typename T>
struct ShapeSampling<T, PST_TRIANGLE, PPM_SOLID_ANGLE>
{
    using scalar_type = T;
    using vector3_type = vector<T, 3>;

    static ShapeSampling<T, PST_TRIANGLE, PPM_SOLID_ANGLE> create(NBL_CONST_REF_ARG(Shape<T, PST_TRIANGLE>) tri)
    {
        ShapeSampling<T, PST_TRIANGLE, PPM_SOLID_ANGLE> retval;
        retval.tri = tri;
        return retval;
    }

    template<typename Ray>
    scalar_type deferredPdf(NBL_CONST_REF_ARG(Ray) ray)
    {
        const vector3_type tri_vertices[3] = {tri.vertex0, tri.vertex1, tri.vertex2};
        shapes::SphericalTriangle<scalar_type> st = shapes::SphericalTriangle<scalar_type>::create(tri_vertices, ray.origin);
        const scalar_type rcpProb = st.solidAngle();
        // if `rcpProb` is NAN then the triangle's solid angle was close to 0.0
        return rcpProb > numeric_limits<scalar_type>::min ? (1.0 / rcpProb) : numeric_limits<scalar_type>::max;
    }

    template<class Aniso>
    vector3_type generate_and_pdf(NBL_REF_ARG(scalar_type) pdf, NBL_REF_ARG(scalar_type) newRayMaxT, NBL_CONST_REF_ARG(vector3_type) origin, NBL_CONST_REF_ARG(Aniso) interaction, bool isBSDF, NBL_CONST_REF_ARG(vector3_type) xi)
    {
        scalar_type rcpPdf;
        const vector3_type tri_vertices[3] = {tri.vertex0, tri.vertex1, tri.vertex2};
        shapes::SphericalTriangle<scalar_type> st = shapes::SphericalTriangle<scalar_type>::create(tri_vertices, origin);
        sampling::SphericalTriangle<scalar_type> sst = sampling::SphericalTriangle<scalar_type>::create(st);

        const vector3_type L = sst.generate(rcpPdf, xi.xy);

        pdf = rcpPdf > numeric_limits<scalar_type>::min ? (1.0 / rcpPdf) : numeric_limits<scalar_type>::max;

        const vector3_type N = tri.getNormalTimesArea();
        newRayMaxT = hlsl::dot<vector3_type>(N, tri.vertex0 - origin) / hlsl::dot<vector3_type>(N, L);
        return L;
    }

    Shape<T, PST_TRIANGLE> tri;
};

template<typename T>
struct ShapeSampling<T, PST_TRIANGLE, PPM_APPROX_PROJECTED_SOLID_ANGLE>
{
    using scalar_type = T;
    using vector3_type = vector<T, 3>;

    static ShapeSampling<T, PST_TRIANGLE, PPM_APPROX_PROJECTED_SOLID_ANGLE> create(NBL_CONST_REF_ARG(Shape<T, PST_TRIANGLE>) tri)
    {
        ShapeSampling<T, PST_TRIANGLE, PPM_APPROX_PROJECTED_SOLID_ANGLE> retval;
        retval.tri = tri;
        return retval;
    }

    template<typename Ray>
    scalar_type deferredPdf(NBL_CONST_REF_ARG(Ray) ray)
    {
        const vector3_type L = ray.direction;
        const vector3_type tri_vertices[3] = {tri.vertex0, tri.vertex1, tri.vertex2};
        shapes::SphericalTriangle<scalar_type> st = shapes::SphericalTriangle<scalar_type>::create(tri_vertices, ray.origin);
        sampling::ProjectedSphericalTriangle<scalar_type> pst = sampling::ProjectedSphericalTriangle<scalar_type>::create(st);
        const scalar_type pdf = pst.backwardPdf(ray.normalAtOrigin, ray.wasBSDFAtOrigin, L);
        // if `pdf` is NAN then the triangle's projected solid angle was close to 0.0, if its close to INF then the triangle was very small
        return pdf < numeric_limits<scalar_type>::max ? pdf : numeric_limits<scalar_type>::max;
    }

    template<class Aniso>
    vector3_type generate_and_pdf(NBL_REF_ARG(scalar_type) pdf, NBL_REF_ARG(scalar_type) newRayMaxT, NBL_CONST_REF_ARG(vector3_type) origin, NBL_CONST_REF_ARG(Aniso) interaction, bool isBSDF, NBL_CONST_REF_ARG(vector3_type) xi)
    {
        scalar_type rcpPdf;
        const vector3_type tri_vertices[3] = {tri.vertex0, tri.vertex1, tri.vertex2};
        shapes::SphericalTriangle<scalar_type> st = shapes::SphericalTriangle<scalar_type>::create(tri_vertices, origin);
        sampling::ProjectedSphericalTriangle<scalar_type> pst = sampling::ProjectedSphericalTriangle<scalar_type>::create(st);

        const vector3_type L = pst.generate(rcpPdf, interaction.getN(), isBSDF, xi.xy);

        pdf = rcpPdf > numeric_limits<scalar_type>::min ? (1.0 / rcpPdf) : numeric_limits<scalar_type>::max;

        const vector3_type N = tri.getNormalTimesArea();
        newRayMaxT = hlsl::dot<vector3_type>(N, tri.vertex0 - origin) / hlsl::dot<vector3_type>(N, L);
        return L;
    }

    Shape<T, PST_TRIANGLE> tri;
};

template<typename T>
struct ShapeSampling<T, PST_RECTANGLE, PPM_AREA>
{
    using scalar_type = T;
    using vector3_type = vector<T, 3>;

    static ShapeSampling<T, PST_RECTANGLE, PPM_AREA> create(NBL_CONST_REF_ARG(Shape<T, PST_RECTANGLE>) rect)
    {
        ShapeSampling<T, PST_RECTANGLE, PPM_AREA> retval;
        retval.rect = rect;
        return retval;
    }

    template<typename Ray>
    scalar_type deferredPdf(NBL_CONST_REF_ARG(Ray) ray)
    {
        const scalar_type dist = ray.intersectionT;
        const vector3_type L = ray.direction;
        return dist * dist / hlsl::abs<scalar_type>(hlsl::dot<vector3_type>(rect.getNormalTimesArea(), L));
    }

    template<class Aniso>
    vector3_type generate_and_pdf(NBL_REF_ARG(scalar_type) pdf, NBL_REF_ARG(scalar_type) newRayMaxT, NBL_CONST_REF_ARG(vector3_type) origin, NBL_CONST_REF_ARG(Aniso) interaction, bool isBSDF, NBL_CONST_REF_ARG(vector3_type) xi)
    {
        const vector3_type N = rect.getNormalTimesArea();
        const vector3_type origin2origin = rect.offset - origin;

        vector3_type L = origin2origin + rect.edge0 * xi.x + rect.edge1 * xi.y;
        const scalar_type distSq = hlsl::dot<vector3_type>(L, L);
        const scalar_type rcpDist = 1.0 / hlsl::sqrt<scalar_type>(distSq);
        L *= rcpDist;
        pdf = distSq / hlsl::abs<scalar_type>(hlsl::dot<vector3_type>(N, L));
        newRayMaxT = 1.0 / rcpDist;
        return L;
    }

    Shape<T, PST_RECTANGLE> rect;
};

template<typename T>
struct ShapeSampling<T, PST_RECTANGLE, PPM_SOLID_ANGLE>
{
    using scalar_type = T;
    using vector3_type = vector<T, 3>;

    static ShapeSampling<T, PST_RECTANGLE, PPM_SOLID_ANGLE> create(NBL_CONST_REF_ARG(Shape<T, PST_RECTANGLE>) rect)
    {
        ShapeSampling<T, PST_RECTANGLE, PPM_SOLID_ANGLE> retval;
        retval.rect = rect;
        return retval;
    }

    template<typename Ray>
    scalar_type deferredPdf(NBL_CONST_REF_ARG(Ray) ray)
    {
        scalar_type pdf;
        matrix<scalar_type, 3, 3> rectNormalBasis;
        vector<T, 2> rectExtents;
        rect.getNormalBasis(rectNormalBasis, rectExtents);
        shapes::SphericalRectangle<scalar_type> sphR0;
        sphR0.origin = rect.offset;
        sphR0.extents = rectExtents;
        sphR0.basis = rectNormalBasis;
        scalar_type solidAngle = sphR0.solidAngle(ray.origin);
        if (solidAngle > numeric_limits<scalar_type>::min)
            pdf = 1.f / solidAngle;
        else
            pdf = bit_cast<scalar_type>(numeric_limits<scalar_type>::infinity);
        return pdf;
    }

    template<class Aniso>
    vector3_type generate_and_pdf(NBL_REF_ARG(scalar_type) pdf, NBL_REF_ARG(scalar_type) newRayMaxT, NBL_CONST_REF_ARG(vector3_type) origin, NBL_CONST_REF_ARG(Aniso) interaction, bool isBSDF, NBL_CONST_REF_ARG(vector3_type) xi)
    {
        const vector3_type N = rect.getNormalTimesArea();
        const vector3_type origin2origin = rect.offset - origin;

        matrix<scalar_type, 3, 3> rectNormalBasis;
        vector<T, 2> rectExtents;
        rect.getNormalBasis(rectNormalBasis, rectExtents);
        shapes::SphericalRectangle<scalar_type> sphR0;
        sphR0.origin = rect.offset;
        sphR0.extents = rectExtents;
        sphR0.basis = rectNormalBasis;
        vector3_type L = hlsl::promote<vector3_type>(0.0);
        scalar_type solidAngle = sphR0.solidAngle(origin);

        sampling::SphericalRectangle<scalar_type> ssph = sampling::SphericalRectangle<scalar_type>::create(sphR0);
        vector<T, 2> sphUv = ssph.generate(origin, xi.xy, solidAngle);
        if (solidAngle > numeric_limits<scalar_type>::min)
        {
            vector3_type sph_sample = sphUv.x * rect.edge0 + sphUv.y * rect.edge1 + rect.offset;
            L = sph_sample - origin;
            const bool invalid = hlsl::all(hlsl::abs(L) < hlsl::promote<vector3_type>(numeric_limits<scalar_type>::min));
            L = hlsl::mix(hlsl::normalize(L), hlsl::promote<vector3_type>(0.0), invalid);
            pdf = hlsl::mix(1.f / solidAngle, bit_cast<scalar_type>(numeric_limits<scalar_type>::infinity), invalid);
        }
        else
            pdf = bit_cast<scalar_type>(numeric_limits<scalar_type>::infinity);

        newRayMaxT = hlsl::dot<vector3_type>(N, origin2origin) / hlsl::dot<vector3_type>(N, L);
        return L;
    }

    Shape<T, PST_RECTANGLE> rect;
};

// PPM_APPROX_PROJECTED_SOLID_ANGLE not available for PST_TRIANGLE


template<class Scene, class Light, typename Ray, class LightSample, class Aniso, IntersectMode Mode, ProceduralShapeType PST, PTPolygonMethod PPM>
struct NextEventEstimator;

template<class Scene, class Light, typename Ray, class LightSample, class Aniso, ProceduralShapeType PST, PTPolygonMethod PPM>
struct NextEventEstimator<Scene, Light, Ray, LightSample, Aniso, IM_PROCEDURAL, PST, PPM>
{
    using scalar_type = typename Ray::scalar_type;
    using vector3_type = vector<scalar_type, 3>;
    using ray_type = Ray;
    using scene_type = Scene;
    using light_type = Light;
    using light_id_type = uint32_t;
    using spectral_type = typename light_type::spectral_type;
    using interaction_type = Aniso;
    using quotient_pdf_type = sampling::quotient_and_pdf<spectral_type, scalar_type>;
    using sample_type = LightSample;
    using ray_dir_info_type = typename sample_type::ray_dir_info_type;

    using shape_type = Shape<scalar_type, PST>;
    using shape_sampling_type = ShapeSampling<scalar_type, PST, PPM>;

    struct SampleQuotientReturn
    {
        sample_type sample_;
        quotient_pdf_type quotient_pdf;
        scalar_type newRayMaxT;
    };
    using sample_quotient_return_type = SampleQuotientReturn;

    template<typename C=bool_constant<PST==PST_SPHERE> NBL_FUNC_REQUIRES(C::value && PST==PST_SPHERE)
    shape_sampling_type __getShapeSampling(uint32_t lightObjectID)
    {
        const shape_type sphere = scene.getSphere(lightObjectID);
        return shape_sampling_type::create(sphere);
    }
    template<typename C=bool_constant<PST==PST_TRIANGLE> NBL_FUNC_REQUIRES(C::value && PST==PST_TRIANGLE)
    shape_sampling_type __getShapeSampling(uint32_t lightObjectID)
    {
        const shape_type tri = scene.getTriangle(lightObjectID);
        return shape_sampling_type::create(tri);
    }
    template<typename C=bool_constant<PST==PST_RECTANGLE> NBL_FUNC_REQUIRES(C::value && PST==PST_RECTANGLE)
    shape_sampling_type __getShapeSampling(uint32_t lightObjectID)
    {
        const shape_type rect = scene.getRectangle(lightObjectID);
        return shape_sampling_type::create(rect);
    }

    scalar_type deferred_pdf(light_id_type lightID, NBL_CONST_REF_ARG(ray_type) ray)
    {
        const light_type light = lights[lightID];
        const shape_sampling_type sampling = __getShapeSampling(light.objectID.id);
        return sampling.template deferredPdf<ray_type>(ray) / scalar_type(lightCount);
    }

    sample_quotient_return_type generate_and_quotient_and_pdf(light_id_type lightID, const spectral_type radiance, const vector3_type origin, NBL_CONST_REF_ARG(interaction_type) interaction, bool isBSDF, const vector3_type xi, uint32_t depth)
    {
        const light_type light = lights[lightID];
        const shape_sampling_type sampling = __getShapeSampling(light.objectID.id);

        sample_quotient_return_type retval;
        scalar_type pdf, newRayMaxT;
        const vector3_type sampleL = sampling.template generate_and_pdf<interaction_type>(pdf, newRayMaxT, origin, interaction, isBSDF, xi);
        ray_dir_info_type rayL;
        if (hlsl::isinf(pdf))
        {
            retval.quotient_pdf = quotient_pdf_type::create(hlsl::promote<spectral_type>(0.0), 0.0);
            retval.sample_ = sample_type::createInvalid();
            return retval;
        }

        const vector3_type N = interaction.getN();
        const scalar_type NdotL = nbl::hlsl::dot<vector3_type>(N, sampleL);
        
        rayL.setDirection(sampleL);
        retval.sample_ = sample_type::create(rayL,interaction.getT(),interaction.getB(),NdotL);

        newRayMaxT *= Tolerance<scalar_type>::getEnd(depth);
        pdf *= 1.0 / scalar_type(lightCount);
        spectral_type quo = radiance / pdf;
        retval.quotient_pdf = quotient_pdf_type::create(quo, pdf);
        retval.newRayMaxT = newRayMaxT;

        return retval;
    }

    spectral_type get_environment_radiance(NBL_CONST_REF_ARG(ray_type) ray)
    {
        // can also sample environment map using ray direction
        return vector3_type(0.15, 0.21, 0.3);
    }

    light_type lights[scene_type::SCENE_LIGHT_COUNT];
    uint32_t lightCount;
    scene_type scene;
};

#endif
