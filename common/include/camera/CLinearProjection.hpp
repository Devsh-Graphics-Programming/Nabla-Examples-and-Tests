#ifndef _NBL_C_LINEAR_PROJECTION_HPP_
#define _NBL_C_LINEAR_PROJECTION_HPP_

#include "ILinearProjection.hpp"

namespace nbl::hlsl
{

class CLinearProjection : public ILinearProjection
{
public:
    using ILinearProjection::ILinearProjection;

	inline static core::smart_refctd_ptr<CLinearProjection> create(core::smart_refctd_ptr<ICamera>&& camera)
	{
		if (!camera)
			return nullptr;

		return core::smart_refctd_ptr<CLinearProjection>(new CLinearProjection(core::smart_refctd_ptr(camera)), core::dont_grab);
	}

private:
	CLinearProjection(core::smart_refctd_ptr<ICamera>&& camera)
		: ILinearProjection(core::smart_refctd_ptr(camera)) {}
	virtual ~CLinearProjection() = default;
};

} // nbl::hlsl namespace

#endif // _NBL_C_LINEAR_PROJECTION_HPP_