#pragma once

#include <nabla.h>

#include "nbl/core/SRange.h"
#include "glm/glm/glm.hpp"
#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl>
#include "Curves.h"
#include "Polyline.h"

#include <nbl/builtin/hlsl/math/equations/cubic.hlsl>
#include <nbl/builtin/hlsl/math/equations/quartic.hlsl>
#include <nbl/builtin/hlsl/shapes/beziers.hlsl>
#include <nbl/ext/TextRendering/TextRendering.h>

using namespace nbl;

enum class HatchFillPattern: uint32_t
{
	SOLID_FILL = 0,
	CHECKERED,
	DIAMONDS,
	CROSS_HATCH,
	HATCH,
	HORIZONTAL,
	VERTICAL,
	INTERWOVEN,
	REVERSE_HATCH,
	SQUARES,
	CIRCLE,
	LIGHT_SHADED,
	SHADED,
	COUNT
};

class Hatch
{
public:
	// this struct will be filled in cpu and sent to gpu for processing as a single DrawObj
	struct CurveHatchBox
	{
		float64_t2 aabbMin;
		float64_t2 aabbMax;
		float32_t2 curveMin[3];
		float32_t2 curveMax[3];
	};

	using bezier_float_t = double;
	using QuadraticBezier = nbl::hlsl::shapes::QuadraticBezier<bezier_float_t>;
	using QuadraticCurve = nbl::hlsl::shapes::Quadratic<bezier_float_t>;

	static std::array<double, 4> bezierBezierIntersections(const QuadraticBezier& bezier, const QuadraticBezier& other);

	static double intersectOrtho(const QuadraticBezier& bezier, double lineConstant, int major);

	// Splits the bezier into segments such that it is now monotonic in the major axis. 
	static bool splitIntoMajorMonotonicSegments(const QuadraticBezier& bezier, std::array<QuadraticBezier, 2>& segments);

	// Assumes the curve is monotonic in major axis, only considers the t = 0, t = 1 and minor axis extremities
	static std::pair<float64_t2, float64_t2> getBezierBoundingBoxMinor(const QuadraticBezier& bezier);
	
	static bool isLineSegment(const QuadraticBezier& bezier);

	class Segment
	{
	public:
		const QuadraticBezier* originalBezier = nullptr;
		// because beziers are broken down,  depending on the type this is t_start or t_end
		double t_start;
		double t_end; // beziers get broken down

		std::array<double, 2> intersect(const Segment& other) const;
		// checks if it's a straight line e.g. if you're sweeping along y axis the it's a line parallel to x
		bool isStraightLineConstantMajor() const;
	};

	Hatch(std::span<CPolyline> lines, const MajorAxis majorAxis, nbl::system::logger_opt_smart_ptr logger = nullptr, int32_t* debugStep = nullptr, const std::function<void(CPolyline, LineStyleInfo)>& debugOutput = {});
	
	// (temporary)
	Hatch(std::vector<CurveHatchBox>&& in_hatchBoxes) :
		hatchBoxes(std::move(in_hatchBoxes))
	{
	};

	const CurveHatchBox& getHatchBox(uint32_t idx) const { return hatchBoxes[idx]; }
	uint32_t getHatchBoxCount() const { return hatchBoxes.size(); }

	// Generate Fill Pattern
	static core::smart_refctd_ptr<asset::ICPUImage> generateHatchFillPatternMSDF(nbl::ext::TextRendering::TextRenderer* textRenderer, HatchFillPattern fillPattern, uint32_t2 msdfExtents);

private:
	std::vector<CurveHatchBox> hatchBoxes;
};

