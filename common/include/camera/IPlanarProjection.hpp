#ifndef _NBL_I_PLANAR_PROJECTION_HPP_
#define _NBL_I_PLANAR_PROJECTION_HPP_

#include "IGimbalBindingLayout.hpp"
#include "ILinearProjection.hpp"

namespace nbl::hlsl
{

/**
* Linear projection wrapper for one camera-facing planar viewport.
*
* The projection owns viewport-local binding layout storage, while runtime input
* processing is expected to happen through `CGimbalInputBinder`.
*/
class IPlanarProjection : public ILinearProjection
{
public:
    //! One perspective or orthographic projection entry plus its viewport-local bindings.
    struct CProjection : public ILinearProjection::CProjection
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
                        base_t::setProjectionMatrix(buildProjectionMatrixPerspectiveFovLH<float64_t>(hlsl::radians(fov), aspectRatio, m_parameters.m_zNear, m_parameters.m_zFar));
                    else
                        base_t::setProjectionMatrix(buildProjectionMatrixPerspectiveFovRH<float64_t>(hlsl::radians(fov), aspectRatio, m_parameters.m_zNear, m_parameters.m_zFar));
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
        inline const CGimbalBindingLayoutStorage& getInputBinding() const { return m_inputBinding; }
        inline CGimbalBindingLayoutStorage& getInputBinding() { return m_inputBinding; }
    private:
        CProjection() = default;
        ProjectionParameters m_parameters;
        CGimbalBindingLayoutStorage m_inputBinding;
    };

protected:
    IPlanarProjection(core::smart_refctd_ptr<ICamera>&& camera)
        : ILinearProjection(core::smart_refctd_ptr(camera)) {}
    virtual ~IPlanarProjection() = default;
};

} // nbl::hlsl namespace

#endif // _NBL_I_PLANAR_PROJECTION_HPP_
