#pragma once

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "nbl/core/SRange.h"
#include "glm/glm/glm.hpp"
#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl>
#include "curves.h"
#include "Polyline.h"

#include <nbl/builtin/hlsl/math/equations/cubic.hlsl>
#include <nbl/builtin/hlsl/math/equations/quartic.hlsl>
#include <nbl/builtin/hlsl/shapes/beziers.hlsl>

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
	using QuadraticEquation = nbl::hlsl::shapes::Quadratic<bezier_float_t>;

	static std::array<double, 4> solveQuarticRoots(double a, double b, double c, double d, double e, double t_start, double t_end);

	static std::array<double, 4> linePossibleIntersections(const QuadraticBezier& bezier, const QuadraticBezier& other);
	static double intersectOrtho(const QuadraticBezier& bezier, double lineConstant, int major);
	static float64_t2 tangent(const QuadraticBezier& bezier, double t);

	// Splits the bezier into segments such that it is now monotonic in the major axis. 
	static bool splitIntoMajorMonotonicSegments(const QuadraticBezier& bezier, std::array<QuadraticBezier, 2>& segments);

	// Assumes the curve is monotonic in major axis, only considers the t = 0, t = 1 and minor axis extremities
	static std::pair<float64_t2, float64_t2> getBezierBoundingBoxMinor(const QuadraticBezier& bezier);

	// Functions for splitting a curve based on t, where 
	// TakeLower gives you the [0, t] range and TakeUpper gives you the [t, 1] range
	static QuadraticBezier splitCurveTakeLower(const QuadraticBezier& bezier, double t);
	static QuadraticBezier splitCurveTakeUpper(const QuadraticBezier& bezier, double t);
	static QuadraticBezier splitCurveRange(const QuadraticBezier& bezier, double left, double right);

	static bool isLineSegment(const QuadraticBezier& bezier);

	class Segment
	{
	public:
		const QuadraticBezier* originalBezier = nullptr;
		// because beziers are broken down,  depending on the type this is t_start or t_end
		double t_start;
		double t_end; // beziers get broken down

		QuadraticBezier getSplitCurve() const;
		std::array<double, 2> intersect(const Segment& other) const;
		// checks if it's a straight line e.g. if you're sweeping along y axis the it's a line parallel to x
		bool isStraightLineConstantMajor() const;
	};
	Hatch(nbl::core::SRange<CPolyline> lines, const MajorAxis majorAxis, int32_t& debugStep, std::function<void(CPolyline, CPULineStyle)> debugOutput /* tmp */);
	// (temporary)
	Hatch(std::vector<CurveHatchBox>&& in_hatchBoxes) :
		hatchBoxes(std::move(in_hatchBoxes))
	{
	};

	const CurveHatchBox& getHatchBox(uint32_t idx) const { return hatchBoxes[idx]; }
	uint32_t getHatchBoxCount() const { return hatchBoxes.size(); }

	std::vector<uint32_t> intersectionAmounts;
private:
	std::vector<CurveHatchBox> hatchBoxes;
};
