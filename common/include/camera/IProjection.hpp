#ifndef _NBL_IPROJECTION_HPP_
#define _NBL_IPROJECTION_HPP_

#include "nbl/builtin/hlsl/cpp_compat/matrix.hlsl"

namespace nbl::hlsl
{

template<typename T>
concept ProjectionMatrix = is_any_of_v<T, float64_t4x4, float32_t4x4, float16_t4x4>;

//! Interface class for projection
template<ProjectionMatrix T>
class IProjection : virtual public core::IReferenceCounted
{
public:
    using value_t = T;

    IProjection(const value_t& matrix = {}) : m_projectionMatrix(matrix) {}
    value_t& getProjectionMatrix() { return m_projectionMatrix; }

    inline bool isLeftHanded()
    {
        return hlsl::determinant(m_projectionMatrix) < 0.f;
    }

protected:
    value_t m_projectionMatrix;
};

template<typename R>
concept ProjectionRange = requires
{
    typename std::ranges::range_value_t<R>;
    // TODO: smart_refctd_ptr check for range_value_t + its type check to grant IProjection<ProjectionMatrix>
};

//! Interface class for a range of IProjection<ProjectionMatrix> projections
template<ProjectionRange Range = std::array<core::smart_refctd_ptr<IProjection<float64_t4x4>>, 1u>>
class IProjectionRange
{
public:
    using range_t = Range;
    using projection_t = std::ranges::range_value_t<range_t>;

    //! Constructor for the range of projections
    IProjectionRange(range_t&& projections) : m_projectionRange(std::move(projections)) {}

    //! Get the stored range of projections
    const range_t& getProjections() const { return m_projectionRange; }

protected:
    range_t m_projectionRange;
};

} // namespace nbl::hlsl

#endif // _NBL_IPROJECTION_HPP_