#ifndef _NBL_I_PLANAR_PROJECTION_HPP_
#define _NBL_I_PLANAR_PROJECTION_HPP_

#include "ILinearProjection.hpp"

namespace nbl::hlsl
{

class IPlanarProjection : public ILinearProjection
{
public:
    struct CProjection : public ILinearProjection::CProjection, public IGimbalController
    {
        using base_t = ILinearProjection::CProjection;

        enum ProjectionType : uint8_t
        {
            Perspective,
            Orthographic,

            Count
        };

        template<ProjectionType T, typename... Args>
        static CProjection create(Args&&... args)
        requires (T != Count)
        {
            CProjection output;

            if constexpr (T == Perspective) output.setPerspective(std::forward<Args>(args)...);
            else if (T == Orthographic) output.setOrthographic(std::forward<Args>(args)...);

            return output;
        }

        CProjection(const CProjection& other) = default;
        CProjection(CProjection&& other) noexcept = default;

        struct ProjectionParameters
        {
            ProjectionType m_type;

            union PlanarParameters
            {
                struct
                {
                    float fov;
                } perspective;

                struct
                {
                    float orthoWidth;
                } orthographic;

                PlanarParameters() {}
                ~PlanarParameters() {}
            } m_planar;

            float m_zNear;
            float m_zFar;
        };

        inline void update(bool leftHanded, float aspectRatio)
        {
            switch (m_parameters.m_type)
            {
                case Perspective:
                {
                    const auto& fov = m_parameters.m_planar.perspective.fov;

                    if (leftHanded)
                        base_t::setProjectionMatrix(buildProjectionMatrixPerspectiveFovLH<float64_t>(glm::radians(fov), aspectRatio, m_parameters.m_zNear, m_parameters.m_zFar));
                    else
                        base_t::setProjectionMatrix(buildProjectionMatrixPerspectiveFovRH<float64_t>(glm::radians(fov), aspectRatio, m_parameters.m_zNear, m_parameters.m_zFar));
                } break;

                case Orthographic:
                {
                    const auto& orthoW = m_parameters.m_planar.orthographic.orthoWidth;
                    const auto viewHeight = orthoW * core::reciprocal(aspectRatio);

                    if (leftHanded)
                        base_t::setProjectionMatrix(buildProjectionMatrixOrthoLH<float64_t>(orthoW, viewHeight, m_parameters.m_zNear, m_parameters.m_zFar));
                    else
                        base_t::setProjectionMatrix(buildProjectionMatrixOrthoRH<float64_t>(orthoW, viewHeight, m_parameters.m_zNear, m_parameters.m_zFar));
                } break;
            }
        }

        inline void setPerspective(float zNear = 0.1f, float zFar = 100.f, float fov = 60.f)
        {
            m_parameters.m_type = Perspective;
            m_parameters.m_planar.perspective.fov = fov;
            m_parameters.m_zNear = zNear;
            m_parameters.m_zFar = zFar;
        }

        inline void setOrthographic(float zNear = 0.1f, float zFar = 100.f, float orthoWidth = 10.f)
        {
            m_parameters.m_type = Orthographic;
            m_parameters.m_planar.orthographic.orthoWidth = orthoWidth;
            m_parameters.m_zNear = zNear;
            m_parameters.m_zFar = zFar;
        }

        inline const ProjectionParameters& getParameters() const { return m_parameters; }
    private:
        CProjection() = default;
        ProjectionParameters m_parameters;
    };

protected:
    IPlanarProjection(core::smart_refctd_ptr<ICamera>&& camera)
        : ILinearProjection(core::smart_refctd_ptr(camera)) {}
    virtual ~IPlanarProjection() = default;
};

} // nbl::hlsl namespace

#endif // _NBL_I_PLANAR_PROJECTION_HPP_