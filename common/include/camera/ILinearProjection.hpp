#ifndef _NBL_I_LINEAR_PROJECTION_HPP_
#define _NBL_I_LINEAR_PROJECTION_HPP_

#include "IProjection.hpp"
#include "ICamera.hpp"

namespace nbl::hlsl
{

//! Interface class for linear projections like Perspective, Orthographic, Oblique, Axonometric, Shear projections or any custom linear transformation - any linear projection represents transform to a viewport
class ILinearProjection
{
public:
    struct CViewportProjection : public IProjection
    {
        using IProjection::IProjection;

        //! underlying type for linear projection matrix type
        using projection_matrix_t = float64_t4x4;

        inline void setProjectionMatrix(const projection_matrix_t& matrix)
        {
            m_projectionMatrix = matrix;
            const auto det = hlsl::determinant(m_projectionMatrix);

            // no singularity for linear projections, but we need to handle it somehow!
            m_isProjectionSingular = not det;

            if (m_isProjectionSingular)
            {
                m_isProjectionLeftHanded = std::nullopt;
                m_invProjectionMatrix = std::nullopt;
            }
            else
            {
                m_isProjectionLeftHanded = det < 0.0;
                m_invProjectionMatrix = inverse(m_projectionMatrix);
            }
        }

        inline const projection_matrix_t& getProjectionMatrix() const { return m_projectionMatrix; }
        inline const std::optional<projection_matrix_t>& getInvProjectionMatrix() const { return m_invProjectionMatrix; }
        inline const std::optional<bool>& isProjectionLeftHanded() const { return m_isProjectionLeftHanded; }
        inline bool isProjectionSingular() const { return m_isProjectionSingular; }
        virtual ProjectionType getProjectionType() const override { return ProjectionType::Linear; }

        virtual void project(const projection_vector_t& vecToProjectionSpace, projection_vector_t& output) const override
        {
            output = mul(m_projectionMatrix, vecToProjectionSpace);
        }

        virtual bool unproject(const projection_vector_t& vecFromProjectionSpace, projection_vector_t& output) const override
        {
            if (m_isProjectionSingular)
                return false;

            output = mul(m_invProjectionMatrix.value(), vecFromProjectionSpace);

            return true;
        }

    private:
        projection_matrix_t m_projectionMatrix;
        std::optional<projection_matrix_t> m_invProjectionMatrix;
        std::optional<bool> m_isProjectionLeftHanded;
        bool m_isProjectionSingular;
    };

    using CProjection = CViewportProjection;

    virtual std::span<const CProjection> getViewportProjections() const = 0;

protected:
    ILinearProjection(core::smart_refctd_ptr<ICamera>&& camera)
        : m_camera(core::smart_refctd_ptr(camera)) {}
    virtual ~ILinearProjection() = default;

    core::smart_refctd_ptr<ICamera> m_camera;
};

} // nbl::hlsl namespace

#endif // _NBL_I_LINEAR_PROJECTION_HPP_