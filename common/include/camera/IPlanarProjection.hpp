#ifndef _NBL_I_PLANAR_PROJECTION_HPP_
#define _NBL_I_PLANAR_PROJECTION_HPP_

#include "ILinearProjection.hpp"

namespace nbl::hlsl
{

class IPlanarProjection : public ILinearProjection
{
public:
    struct CProjection : public ILinearProjection::CProjection, public IGimbalManipulateEncoder
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

        inline void setPerspective(bool leftHanded = true, float zNear = 0.1f, float zFar = 100.f, float fov = 60.f, float aspectRatio = 16.f / 9.f)
        {
            m_parameters.m_type = Perspective;
            m_parameters.m_planar.perspective.fov = fov;
            m_parameters.m_zNear = zNear;
            m_parameters.m_zFar = zFar;

            if (leftHanded)
                base_t::setProjectionMatrix(buildProjectionMatrixPerspectiveFovLH<float64_t>(glm::radians(fov), aspectRatio, zNear, zFar));
            else
                base_t::setProjectionMatrix(buildProjectionMatrixPerspectiveFovRH<float64_t>(glm::radians(fov), aspectRatio, zNear, zFar));
        }

        inline void setOrthographic(bool leftHanded = true, float zNear = 0.1f, float zFar = 100.f, float orthoWidth = 10.f, float aspectRatio = 16.f / 9.f)
        {
            m_parameters.m_type = Orthographic;
            m_parameters.m_planar.orthographic.orthoWidth = orthoWidth;
            m_parameters.m_zNear = zNear;
            m_parameters.m_zFar = zFar;

            const auto viewHeight = orthoWidth * core::reciprocal(aspectRatio);

            if (leftHanded)
                base_t::setProjectionMatrix(buildProjectionMatrixOrthoLH<float64_t>(orthoWidth, viewHeight, zNear, zFar));
            else
                base_t::setProjectionMatrix(buildProjectionMatrixOrthoRH<float64_t>(orthoWidth, viewHeight, zNear, zFar));
        }

        virtual void updateKeyboardMapping(const std::function<void(keyboard_to_virtual_events_t&)>& mapKeys) override { mapKeys(m_keyboardVirtualEventMap); }
        virtual void updateMouseMapping(const std::function<void(mouse_to_virtual_events_t&)>& mapKeys) override { mapKeys(m_mouseVirtualEventMap); }
        virtual void updateImguizmoMapping(const std::function<void(imguizmo_to_virtual_events_t&)>& mapKeys) override { mapKeys(m_imguizmoVirtualEventMap); }

        virtual const keyboard_to_virtual_events_t& getKeyboardVirtualEventMap() const override { return m_keyboardVirtualEventMap; }
        virtual const mouse_to_virtual_events_t& getMouseVirtualEventMap() const override { return m_mouseVirtualEventMap; }
        virtual const imguizmo_to_virtual_events_t& getImguizmoVirtualEventMap() const override { return m_imguizmoVirtualEventMap; }
        inline const ProjectionParameters& getParameters() const { return m_parameters; }
    private:
        CProjection() = default;
        ProjectionParameters m_parameters;

        keyboard_to_virtual_events_t m_keyboardVirtualEventMap;
        mouse_to_virtual_events_t m_mouseVirtualEventMap;
        imguizmo_to_virtual_events_t m_imguizmoVirtualEventMap;
    };

protected:
    IPlanarProjection(core::smart_refctd_ptr<ICamera>&& camera)
        : ILinearProjection(core::smart_refctd_ptr(camera)) {}
    virtual ~IPlanarProjection() = default;
};

} // nbl::hlsl namespace

#endif // _NBL_I_PLANAR_PROJECTION_HPP_