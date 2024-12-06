#ifndef _NBL_I_QUAD_PROJECTION_HPP_
#define _NBL_I_QUAD_PROJECTION_HPP_

#include "ILinearProjection.hpp"

namespace nbl::hlsl
{

/**
* @brief Interface class for quad projections.
*
* This projection transforms a vector into the **model space of a quad**
* (defined by the pre-transform matrix) and then projects it onto the quad using
* the linear view-port transform.
*
* A quad projection is represented by:
* - A **pre-transform matrix** (non-linear/skewed transformation).
* - A **linear view-port transform matrix**.
*
* The final projection matrix is the concatenation of the pre-transform and the linear view-port transform.
*
* @note Single quad projection can represent a face quad of a CAVE-like system.
*/
class IQuadProjection : public ILinearProjection
{
public:
    struct CProjection : ILinearProjection::CProjection
    {
        using base_t = ILinearProjection::CProjection;

        inline void setQuadTransform(const ILinearProjection::model_matrix_t& pretransform, ILinearProjection::concatenated_matrix_t viewport)
        {
            auto concatenated = mul(getMatrix3x4As4x4(pretransform), viewport);
            base_t::setProjectionMatrix(concatenated);

            m_pretransform = pretransform;
            m_viewport = viewport;
        }

    private:
        ILinearProjection::model_matrix_t m_pretransform;
        ILinearProjection::concatenated_matrix_t m_viewport;
    };

protected:
    IQuadProjection(core::smart_refctd_ptr<ICamera>&& camera)
        : ILinearProjection(core::smart_refctd_ptr(camera)) {}
    virtual ~IQuadProjection() = default;
};

} // nbl::hlsl namespace

#endif // _NBL_I_QUAD_PROJECTION_HPP_