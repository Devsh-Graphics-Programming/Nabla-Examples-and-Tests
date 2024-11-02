#ifndef _NBL_IPROJECTION_HPP_
#define _NBL_IPROJECTION_HPP_

#include "nbl/builtin/hlsl/cpp_compat/matrix.hlsl"
#include "IRange.hpp"

namespace nbl::hlsl
{

template<typename T>
concept ProjectionMatrix = is_any_of_v<T, float64_t4x4, float32_t4x4>;

//! Interface class for projection
template<ProjectionMatrix T>
class IProjection : virtual public core::IReferenceCounted
{
public:
    using value_t = T;

    IProjection(const value_t& matrix = {}) : m_projectionMatrix(matrix) { updateHandnessState(); }

    inline void setMatrix(const value_t& projectionMatrix)
    {
        m_projectionMatrix = projectionMatrix;
        updateHandnessState();
    }

    inline const value_t& getMatrix() { return m_projectionMatrix; }
    inline bool isLeftHanded() { return m_isLeftHanded; }

private:
    inline void updateHandnessState()
    {
        m_isLeftHanded = hlsl::determinant(m_projectionMatrix) < 0.f;
    }

    value_t m_projectionMatrix;
    bool m_isLeftHanded;
};

template<typename T>
struct is_projection : std::false_type {};

template<typename T>
struct is_projection<IProjection<T>> : std::true_type {};

template<typename T>
inline constexpr bool is_projection_v = is_projection<T>::value;

template<typename R>
concept ProjectionRange = GeneralPurposeRange<R> && requires
{
    requires core::is_smart_refctd_ptr_v<std::ranges::range_value_t<R>>;
    requires is_projection_v<typename std::ranges::range_value_t<R>::pointee>;
};

//! Interface class for a range of IProjection<ProjectionMatrix> projections
template<ProjectionRange Range = std::array<core::smart_refctd_ptr<IProjection<float64_t4x4>>, 1u>>
class IProjectionRange : public IRange<typename Range>
{
public:
    using base_t = IRange<typename Range>;
    using range_t = typename base_t::range_t;
    using projection_t = typename base_t::range_value_t;

    IProjectionRange(range_t&& projections) : base_t(std::move(projections)) {}
    const range_t& getProjections() const { return base_t::m_range; }
};

} // namespace nbl::hlsl

#endif // _NBL_IPROJECTION_HPP_