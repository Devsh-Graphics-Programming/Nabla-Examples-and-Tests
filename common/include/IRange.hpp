#ifndef _NBL_IRANGE_HPP_
#define _NBL_IRANGE_HPP_

namespace nbl::hlsl
{

template<typename R>
concept GeneralPurposeRange = requires
{
    typename std::ranges::range_value_t<R>;
};

//! Interface class for a general purpose range
template<GeneralPurposeRange Range>
class IRange
{
public:
    using range_t = Range;
    using range_value_t = std::ranges::range_value_t<range_t>;

    IRange(range_t&& range) : m_range(std::move(range)) {}

protected:
    range_t m_range;
};

} // namespace nbl::hlsl

#endif // _NBL_IRANGE_HPP_