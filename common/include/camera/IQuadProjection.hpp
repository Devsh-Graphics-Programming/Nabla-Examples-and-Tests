#ifndef _NBL_I_QUAD_PROJECTION_HPP_
#define _NBL_I_QUAD_PROJECTION_HPP_

#include "ILinearProjection.hpp"

namespace nbl::hlsl
{

//! Interface class for quad projections, basically represents a non-linear/skewed pre-transform concatenated with linear viewport transform, think of it as single quad of a cave designer
class IQuadProjection
{
public:
    struct CCaveFaceProjection : public ILinearProjection::CViewportProjection
    {
        using base_t = ILinearProjection::CViewportProjection;

        //! underlying type for pre-transform projection matrix type
        using pretransform_matrix_t = float64_t3x4;

        inline void setProjectionMatrix(const pretransform_matrix_t& pretransform, const base_t::projection_matrix_t& viewport)
        {
            auto projection = mul(pretransform, viewport);
            base_t::setProjectionMatrix(getMatrix3x4As4x4(projection));
        }

        // TODO: could store "pretransform" & "viewport", may be useful to combine with camera and extract matrices
    };

    using CProjection = CCaveFaceProjection;

    virtual std::span<const CProjection> getQuadProjections() const = 0;

protected:
    IQuadProjection(core::smart_refctd_ptr<ICamera>&& camera)
        : m_camera(core::smart_refctd_ptr(camera)) {}
    virtual ~IQuadProjection() = default;

    core::smart_refctd_ptr<ICamera> m_camera;
};

} // nbl::hlsl namespace

#endif // _NBL_I_QUAD_PROJECTION_HPP_