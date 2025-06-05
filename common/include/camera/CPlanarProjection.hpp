#ifndef _NBL_C_PLANAR_PROJECTION_HPP_
#define _NBL_C_PLANAR_PROJECTION_HPP_

#include "IPlanarProjection.hpp"
#include "IRange.hpp"

namespace nbl::hlsl
{
	template<ContiguousGeneralPurposeRangeOf<IPlanarProjection::CProjection> ProjectionsRange>
	class CPlanarProjection : public IPlanarProjection
	{
	public:
		virtual ~CPlanarProjection() = default;

		inline static core::smart_refctd_ptr<CPlanarProjection> create(core::smart_refctd_ptr<ICamera>&& camera)
		{
			if (!camera)
				return nullptr;

			return core::smart_refctd_ptr<CPlanarProjection>(new CPlanarProjection(core::smart_refctd_ptr(camera)), core::dont_grab);
		}

		virtual std::span<const ILinearProjection::CProjection> getLinearProjections() const override
		{
			return { reinterpret_cast<const ILinearProjection::CProjection*>(m_projections.data()), m_projections.size() };
		}

		inline ProjectionsRange& getPlanarProjections()
		{
			return m_projections;
		}

	protected:
		CPlanarProjection(core::smart_refctd_ptr<ICamera>&& camera)
			: IPlanarProjection(core::smart_refctd_ptr(camera)) {}

		ProjectionsRange m_projections;
	};

} // nbl::hlsl namespace

#endif // _NBL_C_PLANAR_PROJECTION_HPP_