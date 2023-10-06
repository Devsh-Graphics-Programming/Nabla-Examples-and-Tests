
#include "../common/CommonAPI.h"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"
#include "nbl/core/SRange.h"
#include "glm/glm/glm.hpp"
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl>
#include "curves.h"

using namespace nbl;

namespace hatchutils {
	template<class _Ty, size_t _Size>
	struct EquationSolveResult
	{
		size_t uniqueRoots = 0u;
		std::array<_Ty, _Size> roots = {};
	};

    static EquationSolveResult<double, 2> SolveQuadratic(double c[3]);
	static EquationSolveResult<double, 3> SolveCubic(double c[4]);
	static EquationSolveResult<double, 4> SolveQuartic(double c[5]);
    
	static constexpr double QUARTIC_THRESHHOLD = 1e-10;

	static std::array<double, 4> solveQuarticRoots(double a, double b, double c, double d, double e, double t_start, double t_end);
    
	static float64_t2 getCurveRoot(double p0, double p1, double p2);
}

class Hatch
{
public:
	// this struct will be filled in cpu and sent to gpu for processing as a single DrawObj
	struct CurveHatchBox
	{
		float64_t2 aabbMin, aabbMax;
		float64_t2 curveMin[3];
		float64_t2 curveMax[3];
	};

	// TODO: start using A, B, C here
	struct QuadraticBezier {
		float64_t2 p[3];

		std::array<double, 4> linePossibleIntersections(const QuadraticBezier& other) const;
		double intersectOrtho(double coordinate, int major) const;
		float64_t2 evaluateBezier(double t) const;
		float64_t2 tangent(double t) const;
		std::array<double, 4> getRoots() const;
		// Functions for splitting a curve based on t, where 
		// TakeLower gives you the [0, t] range and TakeUpper gives you the [t, 1] range
		QuadraticBezier splitCurveTakeLower(double t) const;
		QuadraticBezier splitCurveTakeUpper(double t) const;
		// Splits the bezier into segments such that it is now monotonic in the major axis. 
		// If it already was major monotonic, ...
		bool splitIntoMajorMonotonicSegments(std::array<QuadraticBezier, 2>& segments) const;
		// Assumes the curve is monotonic in major axis, only considers the t = 0, t = 1 and minor axis extremities
		std::pair<float64_t2, float64_t2> getBezierBoundingBoxMinor() const;
	};

	std::vector<QuadraticBezier> beziers;
	std::vector<CurveHatchBox> hatchBoxes;

	class Segment
	{
	public:
		const QuadraticBezier* originalBezier = nullptr;
		// because beziers are broken down,  depending on the type this is t_start or t_end
		double t_start;
		double t_end; // beziers get broken down

		QuadraticBezier splitCurveRange(QuadraticBezier curve, double left, double right);
		QuadraticBezier getSplitCurve();
		std::array<double, 2> intersect(const Segment& other) const;
		// checks if it's a straight line e.g. if you're sweeping along y axis the it's a line parallel to x
		bool isStraightLineConstantMajor(int major) const;
	};
	Hatch(core::SRange<CPolyline> lines, const MajorAxis majorAxis, std::function<void(CPolyline, CPULineStyle)> debugOutput /* tmp */);
};
