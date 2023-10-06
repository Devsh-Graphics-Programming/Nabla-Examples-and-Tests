#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../common/CommonAPI.h"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"
#include "nbl/core/SRange.h"
#include "glm/glm/glm.hpp"
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl>
#include "curves.h"

static constexpr bool DebugMode = false;
static constexpr bool FragmentShaderPixelInterlock = true;

enum class ExampleMode
{
	CASE_0, // Simple Line, Camera Zoom In/Out
	CASE_1,	// Overdraw Fragment Shader Stress Test
	CASE_2, // hatches
	CASE_3, // CURVES AND LINES
	CASE_4, // STIPPLE PATTERN
};

constexpr ExampleMode mode = ExampleMode::CASE_4;
static constexpr bool DebugMode = true;
static constexpr bool FragmentShaderPixelInterlock = true;

typedef uint32_t uint;

// TODO: Use a math lib?
double dot(const float64_t2& a, const float64_t2& b)
{
	return a.x * b.x + a.y * b.y;
}

double index(const float64_t2& vec, uint32_t index)
{
	const double* arr = &vec.x;
	return arr[index];
}

#include "common.hlsl"

constexpr MajorAxis HatchMajorAxis = MajorAxis::MAJOR_Y;

bool operator==(const LineStyle& lhs, const LineStyle& rhs)
{
	const bool areParametersEqual =
		lhs.color == rhs.color &&
		lhs.screenSpaceLineWidth == rhs.screenSpaceLineWidth &&
		lhs.worldSpaceLineWidth == rhs.worldSpaceLineWidth &&
		lhs.stipplePatternSize == rhs.stipplePatternSize &&
		lhs.recpiprocalStipplePatternLen == rhs.recpiprocalStipplePatternLen &&
		lhs.phaseShift == rhs.phaseShift;

	if (!areParametersEqual)
		return false;
	return true;

	if (lhs.stipplePatternSize == -1)
		return true;

	assert(lhs.stipplePatternSize <= LineStyle::STIPPLE_PATTERN_MAX_SZ && lhs.stipplePatternSize >= 0);

	const bool isStipplePatternArrayEqual = std::memcmp(lhs.stipplePattern, rhs.stipplePattern, sizeof(decltype(lhs.stipplePatternSize)) * lhs.stipplePatternSize);

	return areParametersEqual && isStipplePatternArrayEqual;
}

// holds values for `LineStyle` struct and caculates stipple pattern re values, cant think of better name
class CPULineStyle
{
public:
	static const uint32_t STIPPLE_PATTERN_MAX_SZ = 15u;

	// common data
		// private and setters/getters instead?
	float32_t4 color;
	float screenSpaceLineWidth;
	float worldSpaceLineWidth;

		// TODO[Erfan]: review the logic of this func
	void setStipplePatternData(const nbl::core::SRange<float>& stipplePatternCPURepresentation)
	//void prepareGPUStipplePatternData(const nbl::core::vector<float>& stipplePatternCPURepresentation)
	{
		assert(stipplePatternCPURepresentation.size() <= STIPPLE_PATTERN_MAX_SZ);

		if (stipplePatternCPURepresentation.size() == 0)
		{
			stipplePatternSize = 0;
			return;
		}

		nbl::core::vector<float> stipplePatternTransformed;

		// merge redundant values
		for (auto it = stipplePatternCPURepresentation.begin(); it != stipplePatternCPURepresentation.end();)
		{
			float redundantConsecutiveValuesSum = 0.0f;
			bool isFirstValPositive = (std::abs(*it) == *it);

			do
			{
				redundantConsecutiveValuesSum += *it;
				it++;
			} while (it != stipplePatternCPURepresentation.end() && (isFirstValPositive == (std::abs(*it) == *it)));

			stipplePatternTransformed.push_back(redundantConsecutiveValuesSum);
		}

		if (stipplePatternTransformed.size() == 1)
		{
			stipplePatternSize = stipplePatternTransformed[0] < 0.0f ? -1 : 0;
			return;
		}

		// merge first and last value if their sign matches
		phaseShift = 0.0f;
		bool isFirstComponentNegative = *reinterpret_cast<uint32_t*>(&stipplePatternTransformed[0]) & 0x80000000;
		bool isLastComponentNegative = *reinterpret_cast<uint32_t*>(&stipplePatternTransformed[stipplePatternTransformed.size() - 1]) & 0x80000000;
		if (isFirstComponentNegative == isLastComponentNegative)
		{
			phaseShift = std::abs(stipplePatternTransformed[stipplePatternTransformed.size() - 1]);
			stipplePatternTransformed[0] += stipplePatternTransformed[stipplePatternTransformed.size() - 1];
			stipplePatternTransformed.pop_back();
		}

		if (stipplePatternTransformed.size() == 1)
		{
			stipplePatternSize = isFirstComponentNegative ? -1 : 0;
			return;
		}

		// rotate values if first value is negative value
		if (isFirstComponentNegative)
		{
			std::rotate(stipplePatternTransformed.rbegin(), stipplePatternTransformed.rbegin() + 1, stipplePatternTransformed.rend());
			phaseShift += std::abs(stipplePatternTransformed[0]);
		}

		// calculate normalized prefix sum
		const uint32_t PREFIX_SUM_MAX_SZ = LineStyle::STIPPLE_PATTERN_MAX_SZ - 1u;
		const uint32_t PREFIX_SUM_SZ = stipplePatternTransformed.size() - 1;

		float prefixSum[PREFIX_SUM_MAX_SZ];
		prefixSum[0] = stipplePatternTransformed[0];

		for (uint32_t i = 1u; i < PREFIX_SUM_SZ; i++)
			prefixSum[i] = abs(stipplePatternTransformed[i]) + prefixSum[i - 1];

		auto dbgPatternSize = prefixSum[PREFIX_SUM_SZ - 1] + abs(stipplePatternTransformed[PREFIX_SUM_SZ]);
		recpiprocalStipplePatternLen = 1.0f / (prefixSum[PREFIX_SUM_SZ - 1] + abs(stipplePatternTransformed[PREFIX_SUM_SZ]));

		for (int i = 0; i < PREFIX_SUM_SZ; i++)
			prefixSum[i] *= recpiprocalStipplePatternLen;

		stipplePatternSize = PREFIX_SUM_SZ;
		std::memcpy(stipplePattern, prefixSum, sizeof(prefixSum));

		phaseShift = phaseShift * recpiprocalStipplePatternLen;
	}

	LineStyle getAsGPUData() const
	{
		LineStyle ret;
		std::memcpy(ret.stipplePattern, stipplePattern, STIPPLE_PATTERN_MAX_SZ*sizeof(float));
		ret.color = color;
		ret.screenSpaceLineWidth = screenSpaceLineWidth;
		ret.worldSpaceLineWidth = worldSpaceLineWidth;
		ret.stipplePatternSize = stipplePatternSize;
		ret.recpiprocalStipplePatternLen = recpiprocalStipplePatternLen;
		ret.phaseShift = phaseShift;

		return ret;
	}

	inline bool isVisible() const { return stipplePatternSize != -1; }

	// TODO: private:
public:

	// gpu stipple pattern data form
	int32_t stipplePatternSize = 0;
	float recpiprocalStipplePatternLen;
	float stipplePattern[STIPPLE_PATTERN_MAX_SZ];
	float phaseShift;
};

static_assert(sizeof(DrawObject) == 16u);
static_assert(sizeof(MainObject) == 8u);
static_assert(sizeof(Globals) == 176u);
static_assert(sizeof(ClipProjectionData) == 144u);

using namespace nbl;
using namespace ui;

class Camera2D : public core::IReferenceCounted
{
public:
	Camera2D()
	{}

	void setOrigin(const float64_t2& origin)
	{
		m_origin = origin;
	}

	void setAspectRatio(const double& aspectRatio)
	{
		m_aspectRatio = aspectRatio;
	}

	void setSize(const double size)
	{
		m_bounds = float64_t2{ size * m_aspectRatio, size };
	}

	float64_t2 getBounds() const
	{
		return m_bounds;
	}

	float64_t4x4 constructViewProjection(double timeElapsed)
	{
		auto ret = float64_t4x4();
		//double4x4 ret = {};
		//
		ret[0][0] = 2.0 / m_bounds.x;
		ret[1][1] = -2.0 / m_bounds.y;
		ret[2][2] = 1.0;
		ret[3][3] = 1.0;
		
		ret[0][2] = (-2.0 * m_origin.x) / m_bounds.x;
		ret[1][2] = (2.0 * m_origin.y) / m_bounds.y;

		double theta = 0.0;// (timeElapsed * 0.00008)* (2.0 * nbl::core::PI<double>());

		auto rotation = float64_t4x4(
			cos(theta), -sin(theta), 0.0, 0.0,
			sin(theta), cos(theta), 1.0, 0.0,
			0.0, 0.0, 1.0, 0.0,
			0.0, 0.0, 0.0, 1.0
		);

		float32_t4x4 matrix(rotation * ret);

		return matrix;
	}

	void mouseProcess(const nbl::ui::IMouseEventChannel::range_t& events)
	{
		for (auto eventIt = events.begin(); eventIt != events.end(); eventIt++)
		{
			auto ev = *eventIt;

			if (ev.type == nbl::ui::SMouseEvent::EET_SCROLL)
			{
				m_bounds = m_bounds + float64_t2{ (double)ev.scrollEvent.verticalScroll * -0.1 * m_aspectRatio, (double)ev.scrollEvent.verticalScroll * -0.1};
				m_bounds = float64_t2{ core::max(m_aspectRatio, m_bounds.x), core::max(1.0, m_bounds.y) };
			}
		}
	}

	void keyboardProcess(const IKeyboardEventChannel::range_t& events)
	{
		for (auto eventIt = events.begin(); eventIt != events.end(); eventIt++)
		{
			auto ev = *eventIt;

			if (ev.action == nbl::ui::SKeyboardEvent::E_KEY_ACTION::ECA_PRESSED && ev.keyCode == nbl::ui::E_KEY_CODE::EKC_W)
			{
				m_origin.y += 1;
			}
			if (ev.action == nbl::ui::SKeyboardEvent::E_KEY_ACTION::ECA_PRESSED && ev.keyCode == nbl::ui::E_KEY_CODE::EKC_A)
			{
				m_origin.x -= 1;
			}
			if (ev.action == nbl::ui::SKeyboardEvent::E_KEY_ACTION::ECA_PRESSED && ev.keyCode == nbl::ui::E_KEY_CODE::EKC_S)
			{
				m_origin.y -= 1;
			}
			if (ev.action == nbl::ui::SKeyboardEvent::E_KEY_ACTION::ECA_PRESSED && ev.keyCode == nbl::ui::E_KEY_CODE::EKC_D)
			{
				m_origin.x += 1;
			}
		}
	}
private:

	double m_aspectRatio = 0.0;
	float64_t2 m_bounds = {};
	float64_t2 m_origin = {};
};

// It is not optimized because how you feed a Polyline to our cad renderer is your choice. this is just for convenience
// This is a Nabla Polyline used to feed to our CAD renderer. You can convert your Polyline to this class. or just use it directly.
class CPolyline
{
public:

	// each section consists of multiple connected lines or multiple connected ellipses
	struct SectionInfo
	{
		ObjectType	type;
		uint32_t	index; // can't make this a void* cause of vector resize
		uint32_t	count;
	};

	struct EllipticalArcInfo
	{
		float64_t2 majorAxis;
		float64_t2 center;
		float64_t2 angleBounds; // [0, 2Pi)
		double eccentricity; // (0, 1]

		bool isValid() const
		{
			if (eccentricity > 1.0 || eccentricity < 0.0)
				return false;
			if (angleBounds.y < angleBounds.x)
				return false;
			if ((angleBounds.y - angleBounds.x) > 2 * core::PI<double>())
				return false;
			return true;
		}
	};

	size_t getSectionsCount() const { return m_sections.size(); }

	const SectionInfo& getSectionInfoAt(const uint32_t idx) const
	{
		return m_sections[idx];
	}

	const QuadraticBezierInfo& getQuadBezierInfoAt(const uint32_t idx) const
	{
		return m_quadBeziers[idx];
	}

	const float64_t2& getLinePointAt(const uint32_t idx) const
	{
		return m_linePoints[idx];
	}

	void clearEverything()
	{
		m_sections.clear();
		m_linePoints.clear();
		m_quadBeziers.clear();
	}

	// Reserves memory with worst case
	void reserveMemory(uint32_t noOfLines, uint32_t noOfBeziers)
	{
		m_sections.reserve(noOfLines + noOfBeziers);
		m_linePoints.reserve(noOfLines * 2u);
		m_quadBeziers.reserve(noOfBeziers);
	}

	void addLinePoints(const core::SRange<float64_t2>& linePoints)
	{
		if (linePoints.size() <= 1u)
			return;

		bool addNewSection = m_sections.size() == 0u || m_sections[m_sections.size() - 1u].type != ObjectType::LINE;
		if (addNewSection)
		{
			SectionInfo newSection = {};
			newSection.type = ObjectType::LINE;
			newSection.index = m_linePoints.size();
			newSection.count = linePoints.size() - 1u;
			m_sections.push_back(newSection);
		}
		else
		{
			m_sections[m_sections.size() - 1u].count += linePoints.size();
		}
		m_linePoints.insert(m_linePoints.end(), linePoints.begin(), linePoints.end());
	}

	void addEllipticalArcs(const core::SRange<EllipticalArcInfo>& ellipses)
	{
		// TODO[Erfan] Approximate with quadratic beziers
	}

	// TODO[Przemek]: This uses the struct from the shader common.hlsl if you need to precompute stuff make a duplicate of this struct here first (for the user input to fill)
	// and then do the precomputation here and store in m_quadBeziers which holds the actual structs that will be fed to the GPU
	void addQuadBeziers(const core::SRange<QuadraticBezierInfo>& quadBeziers)
	{
		bool addNewSection = m_sections.size() == 0u || m_sections[m_sections.size() - 1u].type != ObjectType::QUAD_BEZIER;
		if (addNewSection)
		{
			SectionInfo newSection = {};
			newSection.type = ObjectType::QUAD_BEZIER;
			newSection.index = m_quadBeziers.size();
			newSection.count = quadBeziers.size();
			m_sections.push_back(newSection);
		}
		else
		{
			m_sections[m_sections.size() - 1u].count += quadBeziers.size();
		}
		m_quadBeziers.insert(m_quadBeziers.end(), quadBeziers.begin(), quadBeziers.end());
	}

protected:
	std::vector<SectionInfo> m_sections;
	std::vector<float64_t2> m_linePoints;
	std::vector<QuadraticBezierInfo> m_quadBeziers;
};

namespace hatchutils {
	template<class _Ty, size_t _Size>
	struct EquationSolveResult
	{
		size_t uniqueRoots = 0u;
		std::array<_Ty, _Size> roots = {};
	};

	// From https://github.com/erich666/GraphicsGems/blob/master/gems/Roots3And4.c
	// TODO: Refactor this code (needs better names among other things)
	// These are not fully safe on NaNs or when handling precision, as pointed out by devsh in the Discord
	static EquationSolveResult<double, 2> SolveQuadric(double c[3])
	{
		double p, q, D;

		/* normal form: x^2 + px + q = 0 */

		p = c[1] / (2 * c[2]);
		q = c[0] / c[2];

		D = p * p - q;

		if (D == 0.0)
		{
			return { 1, { -p } };
		}
		else if (D < 0)
			return { 0, {} };
		else /* if (D > 0) */
		{
			double sqrt_D = sqrt(D);
			return { 2, { sqrt_D - p, -sqrt_D - p } };
		}
	}

	static EquationSolveResult<double, 3> SolveCubic(double c[4])
	{
		int     i;
		double  sub;
		double  A, B, C;
		double  sq_A, p, q;
		double  cb_p, D;
		EquationSolveResult<double, 3> s;

		/* normal form: x^3 + Ax^2 + Bx + C = 0 */

		A = c[2] / c[3];
		B = c[1] / c[3];
		C = c[0] / c[3];

		/*  substitute x = y - A/3 to eliminate quadric term:
		x^3 +px + q = 0 */

		sq_A = A * A;
		p = 1.0 / 3 * (-1.0 / 3 * sq_A + B);
		q = 1.0 / 2 * (2.0 / 27 * A * sq_A - 1.0 / 3 * A * B + C);

		/* use Cardano's formula */

		cb_p = p * p * p;
		D = q * q + cb_p;

		if (D == 0.0)
		{
			if (q == 0.0) /* one triple solution */
			{
				s = { 1, { 0 } };
			}
			else /* one single and one double solution */
			{
				double u = cbrt(-q);
				s = { 2, { 2 * u, -u } };
			}
		}
		else if (D < 0) /* Casus irreducibilis: three real solutions */
		{
			double phi = 1.0 / 3 * acos(-q / sqrt(-cb_p));
			double t = 2 * sqrt(-p);

			s = { 3, { t * cos(phi), -t * cos(phi + nbl::core::PI<double>() / 3), -t * cos(phi - nbl::core::PI<double>() / 3) } };
		}
		else /* one real solution */
		{
			double sqrt_D = sqrt(D);
			double u = cbrt(sqrt_D - q);
			double v = -cbrt(sqrt_D + q);

			s = { 2, { u + v } };
		}

		/* resubstitute */

		sub = 1.0 / 3 * A;

		for (i = 0; i < s.uniqueRoots; ++i)
			s.roots[i] -= sub;

		return s;
	}


	static EquationSolveResult<double, 4> SolveQuartic(double c[5])
	{
		double  coeffs[4];
		double  z, u, v, sub;
		double  A, B, C, D;
		double  sq_A, p, q, r;
		int     i;
		EquationSolveResult<double, 4> s;

		/* normal form: x^4 + Ax^3 + Bx^2 + Cx + D = 0 */

		A = c[3] / c[4];
		B = c[2] / c[4];
		C = c[1] / c[4];
		D = c[0] / c[4];

		/*  substitute x = y - A/4 to eliminate cubic term:
		x^4 + px^2 + qx + r = 0 */

		sq_A = A * A;
		p = -3.0 / 8 * sq_A + B;
		q = 1.0 / 8 * sq_A * A - 1.0 / 2 * A * B + C;
		r = -3.0 / 256 * sq_A * sq_A + 1.0 / 16 * sq_A * B - 1.0 / 4 * A * C + D;

		if (r == 0.0)
		{
			/* no absolute term: y(y^3 + py + q) = 0 */

			coeffs[0] = q;
			coeffs[1] = p;
			coeffs[2] = 0;
			coeffs[3] = 1;

			auto cubic = SolveCubic(coeffs);
			s = { cubic.uniqueRoots + 1, {} };
			std::copy(cubic.roots.begin(), cubic.roots.end(), s.roots.begin());
			s.roots[cubic.uniqueRoots] = 0;
		}
		else
		{
			/* solve the resolvent cubic ... */

			coeffs[0] = 1.0 / 2 * r * p - 1.0 / 8 * q * q;
			coeffs[1] = -r;
			coeffs[2] = -1.0 / 2 * p;
			coeffs[3] = 1;

			auto cubic = SolveCubic(coeffs);
			s = { cubic.uniqueRoots, {} };
			std::copy(cubic.roots.begin(), cubic.roots.end(), s.roots.begin());

			/* ... and take the one real solution ... */

			z = s.roots[0];

			/* ... to build two quadric equations */

			u = z * z - r;
			v = 2 * z - p;

			if (u == 0.0)
				u = 0;
			else if (u > 0)
				u = sqrt(u);
			else
				return { 0, {} };

			if (v == 0.0)
				v = 0;
			else if (v > 0)
				v = sqrt(v);
			else
				return { 0, {} };

			coeffs[0] = z - u;
			coeffs[1] = q < 0 ? -v : v;
			coeffs[2] = 1;

			auto quadric1 = SolveQuadric(coeffs);

			coeffs[0] = z + u;
			coeffs[1] = q < 0 ? v : -v;
			coeffs[2] = 1;

			auto quadric2 = SolveQuadric(coeffs);

			s = { quadric1.uniqueRoots + quadric2.uniqueRoots, {} };
			std::copy(quadric1.roots.begin(), quadric1.roots.end(), s.roots.begin());
			std::copy(quadric2.roots.begin(), quadric2.roots.end(), s.roots.begin() + quadric1.uniqueRoots);
		}

		/* resubstitute */

		sub = 1.0 / 4 * A;

		for (i = 0; i < s.uniqueRoots; ++i)
			s.roots[i] -= sub;

		return s;
	}

	static constexpr double QUARTIC_THRESHHOLD = 1e-10;

	// Intended to mitigate issues with NaNs and precision by falling back to using simpler functions when the higher roots are small enough
	static std::array<double, 4> solveQuarticRoots(double a, double b, double c, double d, double e, double t_start, double t_end)
	{
		std::array<double, 4> t = { -1.0, -1.0, -1.0, -1.0 }; // only two candidates in range, ever

		const double quadCoeffMag = std::max(std::abs(d), std::abs(e));
		const double cubCoeffMag = std::max(std::abs(c), quadCoeffMag);
		if (std::abs(a) > std::max(std::abs(b), cubCoeffMag) * QUARTIC_THRESHHOLD)
		{
			double params[5] = { e, d, c, b, a };
			auto res = hatchutils::SolveQuartic(params);
			std::copy(res.roots.data(), res.roots.data() + t.size(), t.begin());
		}
		else if (abs(b) > quadCoeffMag)
		{
			double params[4] = { d, c, b, a };
			auto res = hatchutils::SolveCubic(params);
			std::copy(res.roots.data(), res.roots.data() + t.size(), t.begin());
		}
		else
		{
			double params[3] = { c, b, a };
			auto res = hatchutils::SolveQuadric(params);
			std::copy(res.roots.data(), res.roots.data() + t.size(), t.begin());
		}

		if (!(t[0] != t[1]))
			t[0] = t[0] != t_start ? t_start : t_end;

		//// TODO: check that this clamp works with t[i] as NaN
		//for (auto i = 0; i < 2; i++)
		//	t[i] = nbl::core::clamp(t[i], t_start, t_end);
		//
		//// fix up a fuckup where both roots are NaN or were on the same side of the valid integral
		//
		//// TODO: do some Halley or Householder method steps on t
		////while ()
		////{
		////}

		// neither t still not in range, your beziers don't intersect
		return t;
	}

	// (only works for monotonic curves)
	static float64_t2 getCurveRoot(double p0, double p1, double p2)
	{
		double a = p0 - 2.0 * p1 + p2;
		double b = 2.0 * (p1 - p0);
		double c = p0;

		double det = b * b - 4 * a * c;
		double rcp = 0.5 / a;

		double detSqrt = sqrt(det) * rcp;
		double tmp = b * rcp;

		return float64_t2(-detSqrt, detSqrt) - tmp;
	}
};

// Basically 2D CSG
/*
	This class will input a list of Polylines (core::SRange)
	and then output bunch of HatchBoxes
	The hatch box generation algorithm will be used here
*/
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
		QuadraticBezier splitCurveTakeLeft(double t) const;
		QuadraticBezier splitCurveTakeRight(double t) const;
		// Splits the bezier into monotonic segments. If it already was monotonic, 
		// returns a copy of this bezier in the first value of the array
		std::array<QuadraticBezier, 2> splitIntoMonotonicSegments(int major) const;
		std::pair<float64_t2, float64_t2> getBezierBoundingBoxMinor(int major) const;
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

		QuadraticBezier splitCurveRange(QuadraticBezier curve, double left, double right)
		{
			return curve.splitCurveTakeRight(left).splitCurveTakeLeft(right);
		}

		QuadraticBezier getSplitCurve()
		{
			return splitCurveRange(*originalBezier, t_start, t_end);
		}

		// TODO: when we use A, B, C (quadratic coefficients), use this
		//QuadraticBezier getSplitCurve()
		//{
		//	float64_t2 a = originalBezier->p[0] - 2.0 * originalBezier->p[1] + originalBezier->p[2];
		//	float64_t2 b = 2.0 * (originalBezier->p[1] - originalBezier->p[0]);
		//	float64_t2 c = originalBezier->p[0];
		//
		//	return { 
		//		a * (t_end - t_start) * (t_end - t_start),
		//		(t_end - t_start) * (2 * a * t_start + b),
		//		originalBezier->evaluateBezier(t_start)
		//	};
		//}

		std::array<double, 2> intersect(const Segment& other) const
		{
			auto p0 = originalBezier->p[0];
			auto p1 = originalBezier->p[1];
			auto p2 = originalBezier->p[2];
			bool sideP1 = nbl::core::sign((p2.x - p0.x) * (p1.y - p0.y) - (p2.y - p0.y) * (p1.x - p0.x));

			auto otherBezier = *other.originalBezier;
			const std::array<double, 4> intersections = originalBezier->linePossibleIntersections(otherBezier);
			std::array<double, 2> result = { core::nan<double>(), core::nan<double>() };
			int resultIdx = 0;

			for (uint32_t i = 0; i < intersections.size(); i++)
			{
				auto t = intersections[i];
				if (other.t_start >= t || t >= other.t_end)
					continue;

				auto intersection = other.originalBezier->evaluateBezier(t);
				bool sideIntersection = nbl::core::sign<double>((p2.x - p0.x) * (intersection.y - p0.y) - (p2.y - p0.y) * (intersection.x - p0.x));

				// If both P1 and the intersection point are on the same side of the P0 -> P2 line
				// for the current line, we consider this as a valid intersection
				// (Otherwise, continue)
				if (sideP1 != sideIntersection)
					continue;

				result[resultIdx] = t;
				resultIdx++;
			}

			return result;
		}

		// checks if it's a straight line e.g. if you're sweeping along y axis the it's a line parallel to x
		bool isStraightLineConstantMajor(int major) const
		{
			int minor = 1 - major;
			return index(originalBezier->p[0], minor) == index(originalBezier->p[1], minor) && 
				index(originalBezier->p[0], minor) == index(originalBezier->p[2], minor);
		}
	};

	Hatch(core::SRange<CPolyline> lines, const MajorAxis majorAxis, std::function<void(CPolyline, CPULineStyle)> debugOutput /* tmp */)
	{
		std::stack<Segment> starts; // Next segments sorted by start points
		std::stack<Segment> ends; // Next segments sorted by end points
		double maxMajor;

		int major = (int)majorAxis;
		int minor = 1-major; // Minor = Opposite of major (X)

		auto drawDebugBezier = [&](QuadraticBezier bezier, float32_t4 color)
		{
			CPolyline outputPolyline;
			std::vector<QuadraticBezierInfo> beziers;
			QuadraticBezierInfo bezierInfo;
			bezierInfo.p[0] = bezier.p[0];
			bezierInfo.p[1] = bezier.p[1];
			bezierInfo.p[2] = bezier.p[2];
			beziers.push_back(bezierInfo);
			outputPolyline.addQuadBeziers(core::SRange<QuadraticBezierInfo>(beziers.data(), beziers.data() + beziers.size()));

			CPULineStyle cpuLineStyle;
			cpuLineStyle.screenSpaceLineWidth = 1.0f;
			cpuLineStyle.worldSpaceLineWidth = 0.0f;
			cpuLineStyle.color = color;

			debugOutput(outputPolyline, cpuLineStyle);
		};

		auto drawDebugLine = [&](float64_t2 start, float64_t2 end, float32_t4 color)
		{
			CPolyline outputPolyline;
			std::vector<float64_t2> points;
			points.push_back(start);
			points.push_back(end);
			outputPolyline.addLinePoints(core::SRange<float64_t2>(points.data(), points.data() + points.size()));
			
			CPULineStyle cpuLineStyle;
			cpuLineStyle.screenSpaceLineWidth = 1.0f;
			cpuLineStyle.worldSpaceLineWidth = 0.0f;
			cpuLineStyle.color = color;
			
			debugOutput(outputPolyline, cpuLineStyle);
		};

		{
			std::vector<Segment> segments;
			for (CPolyline& line : lines)
			{
				for (uint32_t secIdx = 0; secIdx < line.getSectionsCount(); secIdx ++)
				{
					auto section = line.getSectionInfoAt(secIdx);
					if (section.type == ObjectType::LINE)
						// TODO other types of lines
						{}
					else if (section.type == ObjectType::QUAD_BEZIER)
					{
						for (uint32_t itemIdx = section.index; itemIdx < section.index + section.count; itemIdx ++)
						{
							auto lineBezier = line.getQuadBezierInfoAt(itemIdx);
							QuadraticBezier unsplitBezier = { { lineBezier.p[0], lineBezier.p[1], lineBezier.p[2] }};
							
							// Beziers must be monotonically increasing along major
							// First step: Make sure the bezier is monotonic, split it if not
							auto monotonic = unsplitBezier.splitIntoMonotonicSegments(major);

							auto addBezier = [&](QuadraticBezier bezier)
							{
								auto outputBezier = bezier;
								if (index(outputBezier.evaluateBezier(0.0), major) > index(outputBezier.evaluateBezier(1.0), major))
								{
									outputBezier.p[2] = bezier.p[0];
									outputBezier.p[0] = bezier.p[2];
									assert(index(outputBezier.evaluateBezier(0.0), major) <= index(outputBezier.evaluateBezier(1.0), major));
								}

								beziers.push_back(outputBezier);
							};

							if (nbl::core::isnan(monotonic.data()[0].p[0].x))
							{
								// Already was monotonic
								addBezier(unsplitBezier);
								if (debugOutput)
									drawDebugBezier(unsplitBezier, float32_t4(0.8, 0.8, 0.8, 0.2));
							}
							else
							{
								addBezier(monotonic.data()[0]);
								addBezier(monotonic.data()[1]);
								if (debugOutput)
								{
									drawDebugBezier(monotonic.data()[0], float32_t4(0.0, 0.6, 0.0, 0.5));
									drawDebugBezier(monotonic.data()[1], float32_t4(0.0, 0.0, 0.6, 0.5));
								}
							}
						}
					}
				}
			}
			for (uint32_t bezierIdx = 0; bezierIdx < beziers.size(); bezierIdx++)
			{
				auto hatchBezier = &beziers[bezierIdx];
				Segment segment;
				segment.originalBezier = hatchBezier;
				segment.t_start = 0.0;
				segment.t_end = 1.0;
				segments.push_back(segment);
			}

			// TODO better way to do this
			std::sort(segments.begin(), segments.end(), [&](Segment a, Segment b) { return index(a.originalBezier->p[0], major) > index(b.originalBezier->p[0], major); });
			for (Segment& segment : segments)
				starts.push(segment);

			std::sort(segments.begin(), segments.end(), [&](Segment a, Segment b) { return index(a.originalBezier->p[2], major) > index(b.originalBezier->p[2], major); });
			for (Segment& segment : segments)
				ends.push(segment);
			maxMajor = index(segments.back().originalBezier->p[2], major);
		}

		// Sweep line algorithm
		std::priority_queue<double> intersections; // Next intersection points as major coordinate
		std::vector<Segment> activeCandidates; // Set of active candidates for neighbor search in sweep line

		auto addToCandidateSet = [&](const Segment& entry)
		{
			if (entry.isStraightLineConstantMajor(major))
				return;
			// Look for intersections among active candidates
			// TODO shouldn't this filter out when lines don't intersect?

			// this is a little O(n^2) but only in the `n=candidates.size()`
			for (const auto& segment : activeCandidates)
			{
				// find intersections entry vs segment
				auto intersectionPoints = entry.intersect(segment);
				for (uint32_t i = 0; i < intersectionPoints.size(); i++)
				{
					if (nbl::core::isnan(intersectionPoints[i]))
						continue;
					intersections.push(index(segment.originalBezier->evaluateBezier(intersectionPoints[i]), major));

					if (debugOutput) {
						auto pt = segment.originalBezier->evaluateBezier(intersectionPoints[i]);
						auto min = pt - float64_t2(10.0, 10.0);
						auto max = pt + float64_t2(10.0, 10.0);
						drawDebugLine(float64_t2(min.x, min.y), float64_t2(max.x, min.y), float32_t4(0.0, 0.3, 0.0, 0.1));
						drawDebugLine(float64_t2(max.x, min.y), float64_t2(max.x, max.y), float32_t4(0.0, 0.3, 0.0, 0.1));
						drawDebugLine(float64_t2(min.x, max.y), float64_t2(max.x, max.y), float32_t4(0.0, 0.3, 0.0, 0.1));
						drawDebugLine(float64_t2(min.x, min.y), float64_t2(min.x, max.y), float32_t4(0.0, 0.3, 0.0, 0.1));
					}
				}
			}
			activeCandidates.push_back(entry);
		};

		// if we weren't spawning quads, we could just have unsorted `vector<Bezier*>`
		auto candidateComparator = [&](const Segment& lhs, const Segment& rhs)
		{
			// btw you probably want the beziers in Quadratic At^2+B+C form, not control points
			double _lhs = index(lhs.originalBezier->evaluateBezier(lhs.t_start), major);
			double _rhs = index(rhs.originalBezier->evaluateBezier(lhs.t_start), major);
			if (_lhs == _rhs)
			{
				// this is how you want to order the derivatives dmin/dmaj=-INF dmin/dmaj = 0 dmin/dmaj=INF
				// also leverage the guarantee that `dmaj>=0` to ger numerically stable compare
				float64_t2 lTan = lhs.originalBezier->tangent(lhs.t_start);
				float64_t2 rTan = lhs.originalBezier->tangent(rhs.t_start);
				_lhs = index(lTan, minor) * index(rTan, major);
				_rhs = index(rTan, minor) * index(lTan, major);
				if (_lhs == _rhs)
				{
					// TODO: this is getting the polynominal A for the bezier
					// when bezier gets converted to A, B, C polynominal this is just ->A
					float64_t2 lAcc = lhs.originalBezier->p[0] - 2.0 * lhs.originalBezier->p[1] + lhs.originalBezier->p[2];
					float64_t2 rAcc = lhs.originalBezier->p[0] - 2.0 * lhs.originalBezier->p[1] + lhs.originalBezier->p[2];
					_lhs = index(lAcc, minor) * index(rTan, major);
					_rhs = index(rTan, minor) * index(lAcc, major);
				}
			}
			return _lhs < _rhs;
		};
		auto intersectionVisit = [&]()
		{
			auto newMajor = intersections.top();
			if (debugOutput)
				drawDebugLine(float64_t2(-1000.0, newMajor), float64_t2(1000.0, newMajor), float32_t4(0.3, 1.0, 0.3, 0.1));
			intersections.pop(); // O(n)
			std::cout << "Intersection event at " << newMajor << "\n";
			return newMajor;
		};


		double lastMajor = index(starts.top().originalBezier->evaluateBezier(starts.top().t_start), major);
		std::cout << "\n\nBegin! Max major: " << maxMajor << "\n";
		while (lastMajor!=maxMajor)
		{
			double newMajor;

			const Segment nextStartEvent = starts.empty() ? Segment() : starts.top();
			const double minMajorStart = nextStartEvent.originalBezier ? index(nextStartEvent.originalBezier->evaluateBezier(nextStartEvent.t_start), major) : 0.0;

			const Segment nextEndEvent = ends.top();
			const double maxMajorEnds = index(nextEndEvent.originalBezier->evaluateBezier(nextEndEvent.t_end), major);

			// We check which event, within start, end and intersection events have the smallest
			// major coordinate at this point

			// next start event is before next end event
			if (nextStartEvent.originalBezier && minMajorStart < maxMajorEnds)
			{
				// next start event is before next intersection event
				// (start event)
				if (intersections.empty() || minMajorStart < intersections.top()) // priority queue top() is O(1)
				{
					starts.pop();
					addToCandidateSet(nextStartEvent);
					newMajor = minMajorStart;
					if (debugOutput)
						drawDebugLine(float64_t2(-1000.0, newMajor), float64_t2(1000.0, newMajor), float32_t4(1.0, 0.3, 0.3, 0.1));
					std::cout << "Start event at " << newMajor << "\n";
				}
				// (intersection event)
				else newMajor = intersectionVisit();
			}
			// next end event is before next intersection event
			// (intersection event)
			else if (!intersections.empty() && intersections.top() < maxMajorEnds)
				newMajor = intersectionVisit();
			// (end event)
			else
			{
				newMajor = maxMajorEnds;
				ends.pop();
				if (debugOutput)
					drawDebugLine(float64_t2(-1000.0, newMajor), float64_t2(1000.0, newMajor), float32_t4(0.3, 0.3, 1.0, 0.1));
				std::cout << "End event at " << newMajor << "\n";
			}

			// spawn quads if we advanced
			std::cout << "New major: " << newMajor << " Last major: " << lastMajor << "\n";
			if (newMajor > lastMajor)
			{
				// trim
				const auto candidatesSize = std::distance(activeCandidates.begin(),activeCandidates.end());
				// because n4ce works on loops, this must be true
				assert((candidatesSize % 2u)==0u);
				std::cout << "Candidates size: " << candidatesSize << "\n";
				for (auto i=0u; i< candidatesSize;)
				{
					auto& left = activeCandidates[i++];
					auto& right = activeCandidates[i++];

					CurveHatchBox curveBox;

					auto splitCurveMin = left.getSplitCurve();
					auto splitCurveMax = right.getSplitCurve();

 					auto curveMinAabb = splitCurveMin.getBezierBoundingBoxMinor(major);
					auto curveMaxAabb = splitCurveMax.getBezierBoundingBoxMinor(major);
					curveBox.aabbMin = float64_t2(std::min(curveMinAabb.first.x, curveMaxAabb.first.x), std::min(curveMinAabb.first.y, curveMaxAabb.first.y));
					curveBox.aabbMax = float64_t2(std::max(curveMinAabb.second.x, curveMaxAabb.second.x), std::max(curveMinAabb.second.y, curveMaxAabb.second.y));

					if (debugOutput)
					{
						drawDebugLine(float64_t2(curveBox.aabbMin.x, curveBox.aabbMin.y), float64_t2(curveBox.aabbMax.x, curveBox.aabbMin.y), float32_t4(0.0, 0.3, 0.0, 0.1));
						drawDebugLine(float64_t2(curveBox.aabbMax.x, curveBox.aabbMin.y), float64_t2(curveBox.aabbMax.x, curveBox.aabbMax.y), float32_t4(0.0, 0.3, 0.0, 0.1));
						drawDebugLine(float64_t2(curveBox.aabbMin.x, curveBox.aabbMax.y), float64_t2(curveBox.aabbMax.x, curveBox.aabbMax.y), float32_t4(0.0, 0.3, 0.0, 0.1));
						drawDebugLine(float64_t2(curveBox.aabbMin.x, curveBox.aabbMin.y), float64_t2(curveBox.aabbMin.x, curveBox.aabbMax.y), float32_t4(0.0, 0.3, 0.0, 0.1));
					}
					std::cout << "Hatch box bounding box (" << curveBox.aabbMin.x << ", " << curveBox.aabbMin.y << ") .. (" << curveBox.aabbMax.x << "," << curveBox.aabbMax.y << ")\n";
					// Transform curves into AABB UV space and turn them into quadratic coefficients
					// TODO: the split curve should already have the quadratic bezier as
					// quadratic coefficients
					// so we wont need to convert here
					auto transformCurves = [](Hatch::QuadraticBezier bezier, float64_t2 aabbMin, float64_t2 aabbMax, float64_t2* output) {
						auto rcpAabbExtents = float64_t2(1.0,1.0) / (aabbMax - aabbMin);
						auto p0 = (bezier.p[0] - aabbMin) * rcpAabbExtents;
						auto p1 = (bezier.p[1] - aabbMin) * rcpAabbExtents;
						auto p2 = (bezier.p[2] - aabbMin) * rcpAabbExtents;
						output[0] = p0 - 2.0 * p1 + p2;
						output[1] = 2.0 * (p1 - p0);
						output[2] = p0;
					};
					transformCurves(splitCurveMin, curveBox.aabbMin, curveBox.aabbMax, &curveBox.curveMin[0]);
					transformCurves(splitCurveMax, curveBox.aabbMin, curveBox.aabbMax, &curveBox.curveMax[0]);

					hatchBoxes.push_back(curveBox);
				}

				// advance and trim all of the beziers in the candidate set
				auto oit = activeCandidates.begin();
				for (auto iit = activeCandidates.begin(); iit != activeCandidates.end(); iit++)
				{
					const double evalAtMajor = index(iit->originalBezier->evaluateBezier(iit->t_end), major);
					// if we scrolled past the end of the segment, remove it
					// (basically, we memcpy everything after something is different
					// and we skip on the memcpy for any items that are also different)
					// (this is supposedly a pattern with input/output operators)
					if (newMajor < evalAtMajor)
					{
						const double new_t_start = iit->originalBezier->intersectOrtho(newMajor, major);
						// little optimization (don't memcpy anything before something was removed)
						if (oit != iit)
							*oit = *iit;
						oit->t_start = new_t_start;
						oit++;
					}
				}
				std::sort(activeCandidates.begin(), oit, candidateComparator);
				// trim
				const auto newSize = std::distance(activeCandidates.begin(), oit);
				activeCandidates.resize(newSize);

				lastMajor = newMajor;
			}
		}
	}
private:
};

// returns two possible values of t in the second curve where the curves intersect
std::array<double, 4> Hatch::QuadraticBezier::linePossibleIntersections(const QuadraticBezier& second) const
{
	// Algorithm based on Computer Aided Geometric Design: 
	// https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=1000&context=facpub#page99
	// Chapter 17.6 describes the implicitization of a curve, which transforms it into the following format:
	// Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
	// 
	// We then substitute x and y for the other curve's quadratic formulas.
	// This gives us a quartic equation:
	// At^4 + Bt^3 + Ct^2 + Dt + E = 0
	//
	// The roots for t then become our intersections points between both curves.
	//
	// A Desmos page including math for this as well as some of the graphs it generates is available here:
	// https://www.desmos.com/calculator/mjwqvnvyb8?lang=pt-BR

	double p0x = p[0].x, p1x = p[1].x, p2x = p[2].x,
		p0y = p[0].y, p1y = p[1].y, p2y = p[2].y;

	// Getting the values for the implicitization of the curve
	// TODO: Do this with quadratic coefficients instead (A, B, C)
	double t0 = (4 * p0y * p1y) - (4 * p0y * p2y) - (4 * (p1y * p1y)) + (4 * p1y * p2y) - ((p0y * p0y)) + (2 * p0y * p2y) - ((p2y * p2y));
	double t1 = -(4 * p0x * p1y) + (4 * p0x * p2y) - (4 * p1x * p0y) + (8 * p1x * p1y) - (4 * p1x * p2y) + (4 * p2x * p0y) - (4 * p2x * p1y) + (2 * p0x * p0y) - (2 * p0x * p2y) - (2 * p2x * p0y) + (2 * p2x * p2y);
	double t2 = (4 * p0x * p1x) - (4 * p0x * p2x) - (4 * (p1x * p1x)) + (4 * p1x * p2x) - ((p0x * p0x)) + (2 * p0x * p2x) - ((p2x * p2x));
	double t3 = (4 * p0x * (p1y * p1y)) - (4 * p0x * p1y * p2y) - (4 * p1x * p0y * p1y) + (8 * p1x * p0y * p2y) - (4 * p1x * p1y * p2y) - (4 * p2x * p0y * p1y) + (4 * p2x * (p1y * p1y)) - (2 * p0x * p0y * p2y) + (2 * p0x * (p2y * p2y)) + (2 * p2x * (p0y * p0y)) - (2 * p2x * p0y * p2y);
	double t4 = -(4 * p0x * p1x * p1y) - (4 * p0x * p1x * p2y) + (8 * p0x * p2x * p1y) + (4 * (p1x * p1x) * p0y) + (4 * (p1x * p1x) * p2y) - (4 * p1x * p2x * p0y) - (4 * p1x * p2x * p1y) + (2 * (p0x * p0x) * p2y) - (2 * p0x * p2x * p0y) - (2 * p0x * p2x * p2y) + (2 * (p2x * p2x) * p0y);
	double t5 = (4 * p0x * p1x * p1y * p2y) - (4 * (p1x * p1x) * p0y * p2y) + (4 * p1x * p2x * p0y * p1y) - ((p0x * p0x) * (p2y * p2y)) + (2 * p0x * p2x * p0y * p2y) - ((p2x * p2x) * (p0y * p0y)) - (4 * p0x * p2x * (p1y * p1y));

	// "Slam" the other curve onto it

	float64_t2 A = second.p[0] - 2.0 * second.p[1] + second.p[2];
	float64_t2 B = 2.0 * (second.p[1] - second.p[0]);
	float64_t2 C = second.p[0];

	// Getting the quartic params
	double a = ((A.x * A.x) * t0) + (A.x * A.y * t1) + (A.y * t2);
	double b = (2 * A.x * B.x * t0) + (A.x * B.y * t1) + (B.x * A.y * t1) + (2 * A.y * B.y * t2);
	double c = (2 * A.x * C.x * t0) + (A.x * C.y * t1) + (A.x * t3) + ((B.x * B.x) * t0) + (B.x * B.y * t1) + (C.x * A.y * t1) + (2 * A.y * C.y * t2) + (A.y * t4) + ((B.y * B.y) * t2);
	double d = (2 * B.x * C.x * t0) + (B.x * C.y * t1) + (B.x * t3) + (C.x * B.y * t1) + (2 * B.y * C.y * t2) + (B.y * t4);
	double e = ((C.x * C.x) * t0) + (C.x * C.y * t1) + (C.x * t3) + ((C.y * C.y) * t2) + (C.y * t4) + (t5);

	return hatchutils::solveQuarticRoots(a, b, c, d, e, 0.0, 1.0);
}

double Hatch::QuadraticBezier::intersectOrtho(double coordinate, int major) const
{
	// https://pomax.github.io/bezierinfo/#intersections
	double points[3];
	for (uint32_t i = 0; i < 3; i++)
		points[i] = major ? p[i].y : p[i].x;

	for (uint32_t i = 0; i < 3; i++)
		points[i] -= coordinate;

	float64_t2 roots = hatchutils::getCurveRoot(points[0], points[1], points[2]);
	if (roots.x >= 0.0 && roots.x <= 1.0) return roots.x;
	if (roots.y >= 0.0 && roots.y <= 1.0) return roots.y;
	return core::nan<double>();
}

float64_t2 Hatch::QuadraticBezier::evaluateBezier(double t) const
{
	float64_t2 position = p[0] * (1.0 - t) * (1.0 - t)
		+ 2.0 * p[1] * (1.0 - t) * t
		+ p[2] * t * t;
	return position;
}

// https://pomax.github.io/bezierinfo/#pointvectors
float64_t2 Hatch::QuadraticBezier::tangent(double t) const
{
	// TODO: figure out a tangent algorithm for when this becomes A, B, C
	auto derivativeOrder1First = 2.0 * (p[1] - p[0]);
	auto derivativeOrder1Second = 2.0 * (p[2] - p[1]);
	auto tangent = (1.0 - t) * derivativeOrder1First + t * derivativeOrder1Second;
	auto len = sqrt(tangent.x * tangent.x + tangent.y * tangent.y);
	return tangent / len;
}

// https://pomax.github.io/bezierinfo/#extremities
std::array<double, 4> Hatch::QuadraticBezier::getRoots() const
{
	// Quadratic coefficients
	float64_t2 A = p[0] - 2.0 * p[1] + p[2];
	float64_t2 B = 2.0 * (p[1] - p[0]);
	float64_t2 C = p[0];
	
	auto xroots = hatchutils::getCurveRoot(A.x, B.x, C.x);
	auto yroots = hatchutils::getCurveRoot(A.y, B.y, C.y);

	return { xroots.x, xroots.y, yroots.x, yroots.y };
}

Hatch::QuadraticBezier Hatch::QuadraticBezier::splitCurveTakeLeft(double t) const
{
	QuadraticBezier outputCurve;
	outputCurve.p[0] = p[0];
	outputCurve.p[1] = (1 - t) * p[0] + t * p[1];
	outputCurve.p[2] = (1 - t) * ((1 - t) * p[0] + t * p[1]) + t * ((1 - t) * p[1] + t * p[2]);
	auto p0 = outputCurve.p[0];
	auto p1 = outputCurve.p[1];
	auto p2 = outputCurve.p[2];

	return outputCurve;
}

Hatch::QuadraticBezier Hatch::QuadraticBezier::splitCurveTakeRight(double t) const
{
	QuadraticBezier outputCurve;
	outputCurve.p[0] = p[2];
	outputCurve.p[1] = (1 - t) * p[1] + t * p[2];
	outputCurve.p[2] = (1 - t) * ((1 - t) * p[0] + t * p[1]) + t * ((1 - t) * p[1] + t * p[2]);
	auto p0 = outputCurve.p[0];
	auto p1 = outputCurve.p[1];
	auto p2 = outputCurve.p[2];

	return outputCurve;
}

std::array<Hatch::QuadraticBezier, 2> Hatch::QuadraticBezier::splitIntoMonotonicSegments(int major) const
{
	// Getting derivatives for our quadratic bezier
	auto a = 2.0 * (index(p[1], major) - index(p[0], major));
	auto b = 2.0 * (index(p[2], major) - index(p[1], major));

	// Finding roots for the quadratic bezier derivatives (a straight line)
	auto rcp = 1.0 / (b - a);
	auto t = -a * rcp;
	if (isinf(rcp) || t <= 0.0 || t >= 1.0) return { { float64_t2(nbl::core::nan<double>()) }};
	return { splitCurveTakeLeft(t), splitCurveTakeRight(t) };
}

// https://pomax.github.io/bezierinfo/#boundingbox
std::pair<float64_t2, float64_t2> Hatch::QuadraticBezier::getBezierBoundingBoxMinor(int major) const
{
	int minor = 1 - major;
	double A = index(p[0] - 2.0 * p[1] + p[2], minor);
	double B = index(2.0 * (p[1] - p[0]), minor);

	const int searchTSize = 3;
	double searchT[searchTSize];
	searchT[0] = 0.0;
	searchT[1] = 1.0;
	searchT[2] = -B / (2 * A);

	float64_t2 min = float64_t2(std::numeric_limits<double>::infinity());
	float64_t2 max = float64_t2(-std::numeric_limits<double>::infinity());

	for (uint32_t i = 0; i < searchTSize; i++)
	{
		double t = searchT[i];
		if (t < 0.0 || t > 1.0 || isnan(t))
			continue;
		float64_t2 value = evaluateBezier(t);
		min = float64_t2(std::min(min.x, value.x), std::min(min.y, value.y));
		max = float64_t2(std::max(max.x, value.x), std::max(max.y, value.y));
	}

	return std::pair<float64_t2, float64_t2>(min, max);
}
\
template <typename BufferType>
struct DrawBuffers
{
	core::smart_refctd_ptr<BufferType> indexBuffer;
	core::smart_refctd_ptr<BufferType> mainObjectsBuffer;
	core::smart_refctd_ptr<BufferType> drawObjectsBuffer;
	core::smart_refctd_ptr<BufferType> geometryBuffer;
	core::smart_refctd_ptr<BufferType> lineStylesBuffer;
	core::smart_refctd_ptr<BufferType> customClipProjectionBuffer;
};

// ! this is just a buffers filler with autosubmission features used for convenience to how you feed our CAD renderer
struct DrawBuffersFiller
{
public:

	typedef uint32_t index_buffer_type;

	DrawBuffersFiller() {}

	DrawBuffersFiller(core::smart_refctd_ptr<nbl::video::IUtilities>&& utils)
	{
		utilities = utils;
	}

	typedef std::function<video::IGPUQueue::SSubmitInfo(video::IGPUQueue*, video::IGPUFence*, video::IGPUQueue::SSubmitInfo)> SubmitFunc;

	// function is called when buffer is filled and we should submit draws and clear the buffers and continue filling
	void setSubmitDrawsFunction(SubmitFunc func)
	{
		submitDraws = func;
	}

	void allocateIndexBuffer(core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice, uint32_t indices)
	{
		maxIndices = indices;
		const size_t indexBufferSize = maxIndices * sizeof(uint32_t);

		video::IGPUBuffer::SCreationParams indexBufferCreationParams = {};
		indexBufferCreationParams.size = indexBufferSize;
		indexBufferCreationParams.usage = video::IGPUBuffer::EUF_INDEX_BUFFER_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT;
		gpuDrawBuffers.indexBuffer = logicalDevice->createBuffer(std::move(indexBufferCreationParams));
		gpuDrawBuffers.indexBuffer->setObjectDebugName("indexBuffer");

		video::IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = gpuDrawBuffers.indexBuffer->getMemoryReqs();
		memReq.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
		auto indexBufferMem = logicalDevice->allocate(memReq, gpuDrawBuffers.indexBuffer.get());

		cpuDrawBuffers.indexBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(indexBufferSize);
	}

	void allocateMainObjectsBuffer(core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice, uint32_t mainObjects)
	{
		maxMainObjects = mainObjects;
		size_t mainObjectsBufferSize = mainObjects * sizeof(MainObject);

		video::IGPUBuffer::SCreationParams mainObjectsCreationParams = {};
		mainObjectsCreationParams.size = mainObjectsBufferSize;
		mainObjectsCreationParams.usage = video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT;
		gpuDrawBuffers.mainObjectsBuffer = logicalDevice->createBuffer(std::move(mainObjectsCreationParams));
		gpuDrawBuffers.mainObjectsBuffer->setObjectDebugName("mainObjectsBuffer");

		video::IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = gpuDrawBuffers.mainObjectsBuffer->getMemoryReqs();
		memReq.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
		auto mainObjectsBufferMem = logicalDevice->allocate(memReq, gpuDrawBuffers.mainObjectsBuffer.get());

		cpuDrawBuffers.mainObjectsBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(mainObjectsBufferSize);
	}

	void allocateDrawObjectsBuffer(core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice, uint32_t drawObjects)
	{
		maxDrawObjects = drawObjects;
		size_t drawObjectsBufferSize = drawObjects * sizeof(DrawObject);

		video::IGPUBuffer::SCreationParams drawObjectsCreationParams = {};
		drawObjectsCreationParams.size = drawObjectsBufferSize;
		drawObjectsCreationParams.usage = video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT;
		gpuDrawBuffers.drawObjectsBuffer = logicalDevice->createBuffer(std::move(drawObjectsCreationParams));
		gpuDrawBuffers.drawObjectsBuffer->setObjectDebugName("drawObjectsBuffer");

		video::IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = gpuDrawBuffers.drawObjectsBuffer->getMemoryReqs();
		memReq.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
		auto drawObjectsBufferMem = logicalDevice->allocate(memReq, gpuDrawBuffers.drawObjectsBuffer.get());

		cpuDrawBuffers.drawObjectsBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(drawObjectsBufferSize);
	}

	void allocateGeometryBuffer(core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice, size_t size)
	{
		maxGeometryBufferSize = size;

		video::IGPUBuffer::SCreationParams geometryCreationParams = {};
		geometryCreationParams.size = size;
		geometryCreationParams.usage = core::bitflag(video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT) | video::IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT;
		gpuDrawBuffers.geometryBuffer = logicalDevice->createBuffer(std::move(geometryCreationParams));
		gpuDrawBuffers.geometryBuffer->setObjectDebugName("geometryBuffer");

		video::IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = gpuDrawBuffers.geometryBuffer->getMemoryReqs();
		memReq.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
		auto geometryBufferMem = logicalDevice->allocate(memReq, gpuDrawBuffers.geometryBuffer.get(), video::IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
		geometryBufferAddress = logicalDevice->getBufferDeviceAddress(gpuDrawBuffers.geometryBuffer.get());

		cpuDrawBuffers.geometryBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(size);
	}

	void allocateStylesBuffer(core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice, uint32_t stylesCount)
	{
		maxLineStyles = stylesCount;
		size_t lineStylesBufferSize = stylesCount * sizeof(LineStyle);

		video::IGPUBuffer::SCreationParams lineStylesCreationParams = {};
		lineStylesCreationParams.size = lineStylesBufferSize;
		lineStylesCreationParams.usage = video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT;
		gpuDrawBuffers.lineStylesBuffer = logicalDevice->createBuffer(std::move(lineStylesCreationParams));
		gpuDrawBuffers.lineStylesBuffer->setObjectDebugName("lineStylesBuffer");

		video::IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = gpuDrawBuffers.lineStylesBuffer->getMemoryReqs();
		memReq.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
		auto stylesBufferMem = logicalDevice->allocate(memReq, gpuDrawBuffers.lineStylesBuffer.get());

		cpuDrawBuffers.lineStylesBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(lineStylesBufferSize);
	}

	void allocateCustomClipProjectionBuffer(core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice, uint32_t ClipProjectionDataCount)
	{
		maxClipProjectionData = ClipProjectionDataCount;
		size_t customClipProjectionBufferSize = maxClipProjectionData * sizeof(ClipProjectionData);

		video::IGPUBuffer::SCreationParams customClipProjectionCreationParams = {};
		customClipProjectionCreationParams.size = customClipProjectionBufferSize;
		customClipProjectionCreationParams.usage = video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT;
		gpuDrawBuffers.customClipProjectionBuffer = logicalDevice->createBuffer(std::move(customClipProjectionCreationParams));
		gpuDrawBuffers.customClipProjectionBuffer->setObjectDebugName("customClipProjectionBuffer");

		video::IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = gpuDrawBuffers.customClipProjectionBuffer->getMemoryReqs();
		memReq.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
		auto customClipProjectionBufferMem = logicalDevice->allocate(memReq, gpuDrawBuffers.customClipProjectionBuffer.get());

		cpuDrawBuffers.customClipProjectionBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(customClipProjectionBufferSize);
	}

	uint32_t getIndexCount() const { return currentIndexCount; }
	
	//! this function fills buffers required for drawing a polyline and submits a draw through provided callback when there is not enough memory.
	video::IGPUQueue::SSubmitInfo drawPolyline(
		const CPolyline& polyline,
		const CPULineStyle& cpuLineStyle,
		const uint32_t clipProjectionIdx,
		video::IGPUQueue* submissionQueue,
		video::IGPUFence* submissionFence,
		video::IGPUQueue::SSubmitInfo intendedNextSubmit)
	{
		if (!cpuLineStyle.isVisible())
			return intendedNextSubmit;

		uint32_t styleIdx;
		intendedNextSubmit = addLineStyle_SubmitIfNeeded(cpuLineStyle, styleIdx, submissionQueue, submissionFence, intendedNextSubmit);
		
		MainObject mainObj = {};
		mainObj.styleIdx = styleIdx;
		mainObj.clipProjectionIdx = clipProjectionIdx;
		uint32_t mainObjIdx;
		intendedNextSubmit = addMainObject_SubmitIfNeeded(mainObj, mainObjIdx, submissionQueue, submissionFence, intendedNextSubmit);

		const auto sectionsCount = polyline.getSectionsCount();

		uint32_t currentSectionIdx = 0u;
		uint32_t currentObjectInSection = 0u; // Object here refers to DrawObject used in vertex shader. You can think of it as a Cage.

		while (currentSectionIdx < sectionsCount)
		{
			bool shouldSubmit = false;
			const auto& currentSection = polyline.getSectionInfoAt(currentSectionIdx);
			addPolylineObjects_Internal(polyline, currentSection, currentObjectInSection, mainObjIdx);

			if (currentObjectInSection >= currentSection.count)
			{
				currentSectionIdx++;
				currentObjectInSection = 0u;
			}
			else
				shouldSubmit = true;

			if (shouldSubmit)
			{
				intendedNextSubmit = finalizeAllCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
				intendedNextSubmit = submitDraws(submissionQueue, submissionFence, intendedNextSubmit);
				resetIndexCounters();
				resetGeometryCounters();
				// We don't reset counters for linestyles, mainObjects and customClipProjection because we will be reusing them
				shouldSubmit = false;
			}
		}

		return intendedNextSubmit;
	}

	// If we had infinite mem, we would first upload all curves into geometry buffer then upload the "CurveBoxes" with correct gpu addresses to those
	// But we don't have that so we have to follow a similar auto submission as the "drawPolyline" function with some mutations:
	// We have to find the MAX number of "CurveBoxes" we could draw, and since both the "Curves" and "CurveBoxes" reside in geometry buffer,
	// it has to be taken into account when calculating "how many curve boxes we could draw and when we need to submit/clear"
	// So same as drawPolylines, we would first try to fill the geometry buffer and index buffer that corresponds to "backfaces or even provoking vertices"
	// then change index buffer to draw front faces of the curveBoxes that already reside in geometry buffer memory
	// then if anything was left (the ones that weren't in memory for front face of the curveBoxes) we copy their geom to mem again and use frontface/oddProvoking vertex
	video::IGPUQueue::SSubmitInfo drawHatch(
		const Hatch& hatch,
		const CPULineStyle& lineStyle,
		const uint32_t clipProjectionIdx,
		video::IGPUQueue* submissionQueue,
		video::IGPUFence* submissionFence,
		video::IGPUQueue::SSubmitInfo intendedNextSubmit)
	{
		uint32_t styleIdx;
		intendedNextSubmit = addLineStyle_SubmitIfNeeded(lineStyle, styleIdx, submissionQueue, submissionFence, intendedNextSubmit);
		
		MainObject mainObj = {};
		mainObj.styleIdx = styleIdx;
		mainObj.clipProjectionIdx = clipProjectionIdx;
		uint32_t mainObjIdx;
		intendedNextSubmit = addMainObject_SubmitIfNeeded(mainObj, mainObjIdx, submissionQueue, submissionFence, intendedNextSubmit);

		const auto sectionsCount = 1; //hatch.hatchBoxes.size();

		uint32_t currentSectionIdx = 0u;
		uint32_t currentObjectInSection = 0u; // Object here refers to DrawObject used in vertex shader. You can think of it as a Cage.

		while (currentSectionIdx < sectionsCount)
		{
			bool shouldSubmit = false;
			addHatch_Internal(hatch, currentObjectInSection, mainObjIdx);

			const auto sectionObjectCount = hatch.hatchBoxes.size();
			if (currentObjectInSection >= sectionObjectCount)
			{
				currentSectionIdx++;
				currentObjectInSection = 0u;
			}
			else
				shouldSubmit = true;

			if (shouldSubmit)
			{
				intendedNextSubmit = finalizeAllCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
				intendedNextSubmit = submitDraws(submissionQueue, submissionFence, intendedNextSubmit);
				resetIndexCounters();
				resetGeometryCounters();
				// We don't reset counters for linestyles and mainObjects because we will be reusing them
				shouldSubmit = false;
			}
		}

		return intendedNextSubmit;
	}

	video::IGPUQueue::SSubmitInfo finalizeAllCopiesToGPU(
		video::IGPUQueue* submissionQueue,
		video::IGPUFence* submissionFence,
		video::IGPUQueue::SSubmitInfo intendedNextSubmit)
	{
		intendedNextSubmit = finalizeIndexCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
		intendedNextSubmit = finalizeMainObjectCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
		intendedNextSubmit = finalizeGeometryCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
		intendedNextSubmit = finalizeLineStyleCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
		intendedNextSubmit = finalizeCustomClipProjectionCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);

		return intendedNextSubmit;
	}

	size_t getCurrentIndexBufferSize() const
	{
		return sizeof(index_buffer_type) * currentIndexCount;
	}

	size_t getCurrentMainObjectsBufferSize() const
	{
		return sizeof(MainObject) * currentMainObjectCount;
	}

	size_t getCurrentDrawObjectsBufferSize() const
	{
		return sizeof(DrawObject) * currentDrawObjectCount;
	}

	size_t getCurrentGeometryBufferSize() const
	{
		return currentGeometryBufferSize;
	}

	size_t getCurrentLineStylesBufferSize() const
	{
		return sizeof(LineStyle) * currentLineStylesCount;
	}

	size_t getCurrentCustomClipProjectionBufferSize() const
	{
		return sizeof(ClipProjectionData) * currentClipProjectionDataCount;
	}

	void reset()
	{
		resetAllCounters();
	}

	DrawBuffers<asset::ICPUBuffer> cpuDrawBuffers;
	DrawBuffers<video::IGPUBuffer> gpuDrawBuffers;

protected:

	SubmitFunc submitDraws;
	static constexpr uint32_t InvalidLineStyleIdx = ~0u;

	video::IGPUQueue::SSubmitInfo finalizeIndexCopiesToGPU(
		video::IGPUQueue* submissionQueue,
		video::IGPUFence* submissionFence,
		video::IGPUQueue::SSubmitInfo intendedNextSubmit)
	{
		// Copy Indices
		uint32_t remainingIndexCount = currentIndexCount - inMemIndexCount;
		asset::SBufferRange<video::IGPUBuffer> indicesRange = { sizeof(index_buffer_type) * inMemIndexCount, sizeof(index_buffer_type) * remainingIndexCount, gpuDrawBuffers.indexBuffer };
		const index_buffer_type* srcIndexData = reinterpret_cast<index_buffer_type*>(cpuDrawBuffers.indexBuffer->getPointer()) + inMemIndexCount;
		if (indicesRange.size > 0u)
			intendedNextSubmit = utilities->updateBufferRangeViaStagingBuffer(indicesRange, srcIndexData, submissionQueue, submissionFence, intendedNextSubmit);
		inMemIndexCount = currentIndexCount;
		return intendedNextSubmit;
	}

	video::IGPUQueue::SSubmitInfo finalizeMainObjectCopiesToGPU(
		video::IGPUQueue* submissionQueue,
		video::IGPUFence* submissionFence,
		video::IGPUQueue::SSubmitInfo intendedNextSubmit)
	{
		// Copy MainObjects
		uint32_t remainingMainObjects = currentMainObjectCount - inMemMainObjectCount;
		asset::SBufferRange<video::IGPUBuffer> mainObjectsRange = { sizeof(MainObject) * inMemMainObjectCount, sizeof(MainObject) * remainingMainObjects, gpuDrawBuffers.mainObjectsBuffer };
		const MainObject* srcMainObjData = reinterpret_cast<MainObject*>(cpuDrawBuffers.mainObjectsBuffer->getPointer()) + inMemMainObjectCount;
		if (mainObjectsRange.size > 0u)
			intendedNextSubmit = utilities->updateBufferRangeViaStagingBuffer(mainObjectsRange, srcMainObjData, submissionQueue, submissionFence, intendedNextSubmit);
		inMemMainObjectCount = currentMainObjectCount;
		return intendedNextSubmit;
	}

	video::IGPUQueue::SSubmitInfo finalizeGeometryCopiesToGPU(
		video::IGPUQueue* submissionQueue,
		video::IGPUFence* submissionFence,
		video::IGPUQueue::SSubmitInfo intendedNextSubmit)
	{
		// Copy DrawObjects
		uint32_t remainingDrawObjects = currentDrawObjectCount - inMemDrawObjectCount;
		asset::SBufferRange<video::IGPUBuffer> drawObjectsRange = { sizeof(DrawObject) * inMemDrawObjectCount, sizeof(DrawObject) * remainingDrawObjects, gpuDrawBuffers.drawObjectsBuffer };
		const DrawObject* srcDrawObjData = reinterpret_cast<DrawObject*>(cpuDrawBuffers.drawObjectsBuffer->getPointer()) + inMemDrawObjectCount;
		if (drawObjectsRange.size > 0u)
			intendedNextSubmit = utilities->updateBufferRangeViaStagingBuffer(drawObjectsRange, srcDrawObjData, submissionQueue, submissionFence, intendedNextSubmit);
		inMemDrawObjectCount = currentDrawObjectCount;

		// Copy GeometryBuffer
		uint32_t remainingGeometrySize = currentGeometryBufferSize - inMemGeometryBufferSize;
		asset::SBufferRange<video::IGPUBuffer> geomRange = { inMemGeometryBufferSize, remainingGeometrySize, gpuDrawBuffers.geometryBuffer };
		const uint8_t* srcGeomData = reinterpret_cast<uint8_t*>(cpuDrawBuffers.geometryBuffer->getPointer()) + inMemGeometryBufferSize;
		if (geomRange.size > 0u)
			intendedNextSubmit = utilities->updateBufferRangeViaStagingBuffer(geomRange, srcGeomData, submissionQueue, submissionFence, intendedNextSubmit);
		inMemGeometryBufferSize = currentGeometryBufferSize;

		return intendedNextSubmit;
	}

	video::IGPUQueue::SSubmitInfo finalizeLineStyleCopiesToGPU(
		video::IGPUQueue* submissionQueue,
		video::IGPUFence* submissionFence,
		video::IGPUQueue::SSubmitInfo intendedNextSubmit)
	{
		// Copy LineStyles
		uint32_t remainingLineStyles = currentLineStylesCount - inMemLineStylesCount;
		asset::SBufferRange<video::IGPUBuffer> stylesRange = { sizeof(LineStyle) * inMemLineStylesCount, sizeof(LineStyle) * remainingLineStyles, gpuDrawBuffers.lineStylesBuffer };
		const LineStyle* srcLineStylesData = reinterpret_cast<LineStyle*>(cpuDrawBuffers.lineStylesBuffer->getPointer()) + inMemLineStylesCount;
		if (stylesRange.size > 0u)
			intendedNextSubmit = utilities->updateBufferRangeViaStagingBuffer(stylesRange, srcLineStylesData, submissionQueue, submissionFence, intendedNextSubmit);
		inMemLineStylesCount = currentLineStylesCount;
		return intendedNextSubmit;
	}

	video::IGPUQueue::SSubmitInfo finalizeCustomClipProjectionCopiesToGPU(
		video::IGPUQueue* submissionQueue,
		video::IGPUFence* submissionFence,
		video::IGPUQueue::SSubmitInfo intendedNextSubmit)
	{
		// Copy LineStyles
		uint32_t remainingClipProjectionData = currentClipProjectionDataCount - inMemClipProjectionDataCount;
		asset::SBufferRange<video::IGPUBuffer> clipProjectionRange = { sizeof(ClipProjectionData) * inMemClipProjectionDataCount, sizeof(ClipProjectionData) * remainingClipProjectionData, gpuDrawBuffers.customClipProjectionBuffer };
		const ClipProjectionData* srcClipProjectionData = reinterpret_cast<ClipProjectionData*>(cpuDrawBuffers.customClipProjectionBuffer->getPointer()) + inMemClipProjectionDataCount;
		if (clipProjectionRange.size > 0u)
			intendedNextSubmit = utilities->updateBufferRangeViaStagingBuffer(clipProjectionRange, srcClipProjectionData, submissionQueue, submissionFence, intendedNextSubmit);
		inMemClipProjectionDataCount = currentClipProjectionDataCount;
		return intendedNextSubmit;
	}

	video::IGPUQueue::SSubmitInfo addMainObject_SubmitIfNeeded(
		const MainObject& mainObject,
		uint32_t& outMainObjectIdx,
		video::IGPUQueue* submissionQueue,
		video::IGPUFence* submissionFence,
		video::IGPUQueue::SSubmitInfo intendedNextSubmit)
	{
		outMainObjectIdx = addMainObject_Internal(mainObject);
		if (outMainObjectIdx == InvalidMainObjectIdx)
		{
			intendedNextSubmit = finalizeAllCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
			intendedNextSubmit = submitDraws(submissionQueue, submissionFence, intendedNextSubmit);
			resetAllCounters();
			outMainObjectIdx = addMainObject_Internal(mainObject);
			assert(outMainObjectIdx != InvalidMainObjectIdx);
		}
		return intendedNextSubmit;
	}

	uint32_t addMainObject_Internal(const MainObject& mainObject)
	{
		MainObject* mainObjsArray = reinterpret_cast<MainObject*>(cpuDrawBuffers.mainObjectsBuffer->getPointer());
		if (currentMainObjectCount >= maxMainObjects)
			return InvalidMainObjectIdx;

		void* dst = mainObjsArray + currentMainObjectCount;
		memcpy(dst, &mainObject, sizeof(MainObject));
		uint32_t ret = (currentMainObjectCount % MaxIndexableMainObjects); // just to wrap around if it ever exceeded (we pack this id into 24 bits)
		currentMainObjectCount++;
		return ret;
	}

	video::IGPUQueue::SSubmitInfo addLineStyle_SubmitIfNeeded(
		const CPULineStyle& cpuLineStyle,
		uint32_t& outLineStyleIdx,
		video::IGPUQueue* submissionQueue,
		video::IGPUFence* submissionFence,
		video::IGPUQueue::SSubmitInfo intendedNextSubmit)
	{
		outLineStyleIdx = addLineStyle_Internal(cpuLineStyle);
		if (outLineStyleIdx == InvalidLineStyleIdx)
		{
			intendedNextSubmit = finalizeAllCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
			intendedNextSubmit = submitDraws(submissionQueue, submissionFence, intendedNextSubmit);
			resetAllCounters();
			outLineStyleIdx = addLineStyle_Internal(cpuLineStyle);
			assert(outLineStyleIdx != InvalidLineStyleIdx);
		}
		return intendedNextSubmit;
	}

	uint32_t addLineStyle_Internal(const CPULineStyle& cpuLineStyle)
	{
		LineStyle gpuLineStyle = cpuLineStyle.getAsGPUData();
		LineStyle* stylesArray = reinterpret_cast<LineStyle*>(cpuDrawBuffers.lineStylesBuffer->getPointer());
		for (uint32_t i = 0u; i < currentLineStylesCount; ++i)
		{
			const LineStyle& itr = stylesArray[i];

			if(itr == gpuLineStyle)
				return i;
		}

		if (currentLineStylesCount >= maxLineStyles)
			return InvalidLineStyleIdx;

		void* dst = stylesArray + currentLineStylesCount;
		memcpy(dst, &gpuLineStyle, sizeof(LineStyle));
		return currentLineStylesCount++;
	}

public:
	video::IGPUQueue::SSubmitInfo addClipProjectionData_SubmitIfNeeded(
		const ClipProjectionData& clipProjectionData,
		uint32_t& outClipProjectionIdx,
		video::IGPUQueue* submissionQueue,
		video::IGPUFence* submissionFence,
		video::IGPUQueue::SSubmitInfo intendedNextSubmit)
	{
		outClipProjectionIdx = addClipProjectionData_Internal(clipProjectionData);
		if (outClipProjectionIdx == InvalidClipProjectionIdx)
		{
			intendedNextSubmit = finalizeAllCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
			intendedNextSubmit = submitDraws(submissionQueue, submissionFence, intendedNextSubmit);
			resetAllCounters();
			outClipProjectionIdx = addClipProjectionData_Internal(clipProjectionData);
			assert(outClipProjectionIdx != InvalidClipProjectionIdx);
		}
		return intendedNextSubmit;
	}

protected:
	uint32_t addClipProjectionData_Internal(const ClipProjectionData& clipProjectionData)
	{
		ClipProjectionData* clipProjectionArray = reinterpret_cast<ClipProjectionData*>(cpuDrawBuffers.customClipProjectionBuffer->getPointer());
		if (currentClipProjectionDataCount >= maxClipProjectionData)
			return InvalidClipProjectionIdx;

		void* dst = clipProjectionArray + currentClipProjectionDataCount;
		memcpy(dst, &clipProjectionData, sizeof(ClipProjectionData));
		return currentClipProjectionDataCount++;
	}

	static constexpr uint32_t getCageCountPerPolylineObject(ObjectType type)
	{
		if (type == ObjectType::LINE)
			return 1u;
		else if (type == ObjectType::QUAD_BEZIER)
			return 3u;
		return 0u;
	};

	void addPolylineObjects_Internal(const CPolyline& polyline, const CPolyline::SectionInfo& section, uint32_t& currentObjectInSection, uint32_t mainObjIdx)
	{
		if (section.type == ObjectType::LINE)
			addLines_Internal(polyline, section, currentObjectInSection, mainObjIdx);
		else if (section.type == ObjectType::QUAD_BEZIER)
			addQuadBeziers_Internal(polyline, section, currentObjectInSection, mainObjIdx);
		else
			assert(false); // we don't handle other object types
	}

	void addLines_Internal(const CPolyline& polyline, const CPolyline::SectionInfo& section, uint32_t& currentObjectInSection, uint32_t mainObjIdx)
	{
		assert(section.count >= 1u);
		assert(section.type == ObjectType::LINE);

		const auto maxGeometryBufferPoints = (maxGeometryBufferSize - currentGeometryBufferSize) / sizeof(float64_t2);
		const auto maxGeometryBufferLines = (maxGeometryBufferPoints <= 1u) ? 0u : maxGeometryBufferPoints - 1u;

		uint32_t uploadableObjects = (maxIndices - currentIndexCount) / 6u;
		uploadableObjects = core::min(uploadableObjects, maxGeometryBufferLines);
		uploadableObjects = core::min(uploadableObjects, maxDrawObjects - currentDrawObjectCount);

		const auto lineCount = section.count;
		const auto remainingObjects = lineCount - currentObjectInSection;
		uint32_t objectsToUpload = core::min(uploadableObjects, remainingObjects);

		// Add Indices
		addPolylineObjectIndices_Internal(currentDrawObjectCount, objectsToUpload);

		// Add DrawObjs
		DrawObject drawObj = {};
		drawObj.mainObjIndex = mainObjIdx;
		drawObj.type_subsectionIdx = uint32_t(static_cast<uint16_t>(ObjectType::LINE) | 0 << 16);
		drawObj.geometryAddress = geometryBufferAddress + currentGeometryBufferSize;
		for (uint32_t i = 0u; i < objectsToUpload; ++i)
		{
			void* dst = reinterpret_cast<DrawObject*>(cpuDrawBuffers.drawObjectsBuffer->getPointer()) + currentDrawObjectCount;
			memcpy(dst, &drawObj, sizeof(DrawObject));
			currentDrawObjectCount += 1u;
			drawObj.geometryAddress += sizeof(float64_t2);
		}

		// Add Geometry
		if (objectsToUpload > 0u)
		{
			const auto pointsByteSize = sizeof(float64_t2) * (objectsToUpload + 1u);
			void* dst = reinterpret_cast<char*>(cpuDrawBuffers.geometryBuffer->getPointer()) + currentGeometryBufferSize;
			auto& linePoint = polyline.getLinePointAt(section.index + currentObjectInSection);
			memcpy(dst, &linePoint, pointsByteSize);
			currentGeometryBufferSize += pointsByteSize;
		}

		currentObjectInSection += objectsToUpload;
	}

	void addQuadBeziers_Internal(const CPolyline& polyline, const CPolyline::SectionInfo& section, uint32_t& currentObjectInSection, uint32_t mainObjIdx)
	{
		constexpr uint32_t CagesPerQuadBezier = getCageCountPerPolylineObject(ObjectType::QUAD_BEZIER);
		constexpr uint32_t IndicesPerQuadBezier = 6u * CagesPerQuadBezier;
		assert(section.type == ObjectType::QUAD_BEZIER);

		const auto maxGeometryBufferEllipses = (maxGeometryBufferSize - currentGeometryBufferSize) / sizeof(QuadraticBezierInfo);

		uint32_t uploadableObjects = (maxIndices - currentIndexCount) / IndicesPerQuadBezier;
		uploadableObjects = core::min(uploadableObjects, maxGeometryBufferEllipses);
		uploadableObjects = core::min(uploadableObjects, maxDrawObjects - currentDrawObjectCount);

		const auto beziersCount = section.count;
		const auto remainingObjects = beziersCount - currentObjectInSection;
		uint32_t objectsToUpload = core::min(uploadableObjects, remainingObjects);

		// Add Indices
		addPolylineObjectIndices_Internal(currentDrawObjectCount, objectsToUpload * CagesPerQuadBezier);

		// Add DrawObjs
		DrawObject drawObj = {};
		drawObj.mainObjIndex = mainObjIdx;
		drawObj.geometryAddress = geometryBufferAddress + currentGeometryBufferSize;
		for (uint32_t i = 0u; i < objectsToUpload; ++i)
		{
			for (uint16_t subObject = 0; subObject < CagesPerQuadBezier; subObject++)
			{
				drawObj.type_subsectionIdx = uint32_t(static_cast<uint16_t>(ObjectType::QUAD_BEZIER) | (subObject << 16));
				void* dst = reinterpret_cast<DrawObject*>(cpuDrawBuffers.drawObjectsBuffer->getPointer()) + currentDrawObjectCount;
				memcpy(dst, &drawObj, sizeof(DrawObject));
				currentDrawObjectCount += 1u;
			}
			drawObj.geometryAddress += sizeof(QuadraticBezierInfo);
		}

		// Add Geometry
		if (objectsToUpload > 0u)
		{
			const auto beziersByteSize = sizeof(QuadraticBezierInfo) * (objectsToUpload);
			void* dst = reinterpret_cast<char*>(cpuDrawBuffers.geometryBuffer->getPointer()) + currentGeometryBufferSize;
			auto& quadBezier = polyline.getQuadBezierInfoAt(section.index + currentObjectInSection);
			memcpy(dst, &quadBezier, beziersByteSize);
			currentGeometryBufferSize += beziersByteSize;
		}

		currentObjectInSection += objectsToUpload;
	}

	void addHatch_Internal(const Hatch& hatch, uint32_t& currentObjectInSection, uint32_t mainObjIndex)
	{
		constexpr uint32_t IndicesPerHatchBox = 6u;
		uint32_t uploadableObjects = (maxIndices - currentIndexCount) / IndicesPerHatchBox;
		uploadableObjects = core::min(uploadableObjects, maxDrawObjects - currentDrawObjectCount);
		uploadableObjects = core::min(uploadableObjects, maxGeometryBufferSize - currentGeometryBufferSize);

		uint32_t i = 0;
		for (; i + currentObjectInSection < hatch.hatchBoxes.size() && i < uploadableObjects; i++)
		{
			Hatch::CurveHatchBox hatchBox = hatch.hatchBoxes[i + currentObjectInSection];

			uint64_t hatchBoxAddress;
			{
				CurveBox curveBox;
				curveBox.aabbMin = hatchBox.aabbMin;
				curveBox.aabbMax = hatchBox.aabbMax;
				memcpy(&curveBox.curveMin[0], &hatchBox.curveMin[0], sizeof(float64_t2) * 3);
				memcpy(&curveBox.curveMax[0], &hatchBox.curveMax[0], sizeof(float64_t2) * 3);

				void* dst = reinterpret_cast<char*>(cpuDrawBuffers.geometryBuffer->getPointer()) + currentGeometryBufferSize;
				memcpy(dst, &curveBox, sizeof(CurveBox));
				hatchBoxAddress = geometryBufferAddress + currentGeometryBufferSize;
				currentGeometryBufferSize += sizeof(CurveBox);
			}

			DrawObject drawObj = {};
			drawObj.type_subsectionIdx = uint32_t(static_cast<uint16_t>(ObjectType::CURVE_BOX) | (0 << 16));
			drawObj.mainObjIndex = mainObjIndex;
			drawObj.geometryAddress = hatchBoxAddress;
			void* dst = reinterpret_cast<DrawObject*>(cpuDrawBuffers.drawObjectsBuffer->getPointer()) + currentDrawObjectCount + i;
			memcpy(dst, &drawObj, sizeof(DrawObject));
		}

		// Add Indices
		addHatchIndices_Internal(currentDrawObjectCount, i);
		currentDrawObjectCount += i;
		currentObjectInSection += i;
	}

	//@param oddProvokingVertex is used for our polyline-wide transparency algorithm where we draw the object twice, once to resolve the alpha and another time to draw them
	void addPolylineObjectIndices_Internal(uint32_t startObject, uint32_t objectCount)
	{
		constexpr bool oddProvokingVertex = true; // was useful before, might probably deprecate it later for simplicity or it might be useful for some tricks later on
		index_buffer_type* indices = reinterpret_cast<index_buffer_type*>(cpuDrawBuffers.indexBuffer->getPointer()) + currentIndexCount;
		for (uint32_t i = 0u; i < objectCount; ++i)
		{
			index_buffer_type objIndex = startObject + i;
			if (oddProvokingVertex)
			{
				indices[i * 6] = objIndex * 4u + 1u;
				indices[i * 6 + 1u] = objIndex * 4u + 0u;
			}
			else
			{
				indices[i * 6] = objIndex * 4u + 0u;
				indices[i * 6 + 1u] = objIndex * 4u + 1u;
			}
			indices[i * 6 + 2u] = objIndex * 4u + 2u;

			if (oddProvokingVertex)
			{
				indices[i * 6 + 3u] = objIndex * 4u + 1u;
				indices[i * 6 + 4u] = objIndex * 4u + 2u;
			}
			else
			{
				indices[i * 6 + 3u] = objIndex * 4u + 2u;
				indices[i * 6 + 4u] = objIndex * 4u + 1u;
			}
			indices[i * 6 + 5u] = objIndex * 4u + 3u;
		}
		currentIndexCount += objectCount * 6u;
	}

	void addHatchIndices_Internal(uint32_t startObject, uint32_t objectCount)
	{
		index_buffer_type* indices = reinterpret_cast<index_buffer_type*>(cpuDrawBuffers.indexBuffer->getPointer()) + currentIndexCount;

		for (uint32_t i = 0u; i < objectCount; ++i)
		{
			index_buffer_type objIndex = startObject + i;
			indices[i * 6 + 0u] = objIndex * 4u;
			indices[i * 6 + 1u] = objIndex * 4u + 1u;
			indices[i * 6 + 2u] = objIndex * 4u + 2u;
			indices[i * 6 + 3u] = objIndex * 4u + 1u;
			indices[i * 6 + 4u] = objIndex * 4u + 2u;
			indices[i * 6 + 5u] = objIndex * 4u + 3u;
		}
		currentIndexCount += objectCount * 6u;
	}

	void resetAllCounters()
	{
		resetMainObjectCounters();
		resetGeometryCounters();
		resetIndexCounters();
		resetStyleCounters();
		resetCustomClipProjectionCounters();
	}

	void resetMainObjectCounters()
	{
		inMemMainObjectCount = 0u;
		currentMainObjectCount = 0u;
	}

	void resetGeometryCounters()
	{
		inMemDrawObjectCount = 0u;
		currentDrawObjectCount = 0u;

		inMemGeometryBufferSize = 0u;
		currentGeometryBufferSize = 0u;
	}

	void resetIndexCounters()
	{
		inMemIndexCount = 0u;
		currentIndexCount = 0u;
	}

	void resetStyleCounters()
	{
		currentLineStylesCount = 0u;
		inMemLineStylesCount = 0u;
	}
	
	void resetCustomClipProjectionCounters()
	{
		currentClipProjectionDataCount = 0u;
		inMemClipProjectionDataCount = 0u;
	}

	core::smart_refctd_ptr<nbl::video::IUtilities> utilities;

	uint32_t inMemIndexCount = 0u;
	uint32_t currentIndexCount = 0u;
	uint32_t maxIndices = 0u;

	uint32_t inMemMainObjectCount = 0u;
	uint32_t currentMainObjectCount = 0u;
	uint32_t maxMainObjects = 0u;

	uint32_t inMemDrawObjectCount = 0u;
	uint32_t currentDrawObjectCount = 0u;
	uint32_t maxDrawObjects = 0u;

	uint64_t inMemGeometryBufferSize = 0u;
	uint64_t currentGeometryBufferSize = 0u;
	uint64_t maxGeometryBufferSize = 0u;

	uint32_t inMemLineStylesCount = 0u;
	uint32_t currentLineStylesCount = 0u;
	uint32_t maxLineStyles = 0u;

	uint32_t inMemClipProjectionDataCount = 0u;
	uint32_t currentClipProjectionDataCount = 0u;
	uint32_t maxClipProjectionData = 0u;

	uint64_t geometryBufferAddress = 0u; // Actual BDA offset 0 of the gpu buffer
};

class CADApp : public ApplicationBase
{
	constexpr static uint32_t FRAMES_IN_FLIGHT = 3u;
	static constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;

	constexpr static uint32_t WIN_W = 1600u;
	constexpr static uint32_t WIN_H = 720u;

	CommonAPI::InputSystem::ChannelReader<IMouseEventChannel> mouse;
	CommonAPI::InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

	core::smart_refctd_ptr<video::IQueryPool> pipelineStatsPool;
	core::smart_refctd_ptr<nbl::ui::IWindowManager> windowManager;
	core::smart_refctd_ptr<nbl::ui::IWindow> window;
	core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> windowCb;
	core::smart_refctd_ptr<nbl::video::IAPIConnection> apiConnection;
	core::smart_refctd_ptr<nbl::video::ISurface> surface;
	core::smart_refctd_ptr<nbl::video::IUtilities> utilities;
	core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice;
	video::IPhysicalDevice* physicalDevice;
	std::array<video::IGPUQueue*, CommonAPI::InitOutput::MaxQueuesCount> queues;
	core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
	core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpassInitial; // this renderpass will clear the attachment and transition it to COLOR_ATTACHMENT_OPTIMAL
	core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpassInBetween; // this renderpass will load the attachment and transition it to COLOR_ATTACHMENT_OPTIMAL
	core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpassFinal; // this renderpass will load the attachment and transition it to PRESENT
	nbl::core::smart_refctd_dynamic_array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>> framebuffersDynArraySmartPtr;
	std::array<std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool>, CommonAPI::InitOutput::MaxFramesInFlight>, CommonAPI::InitOutput::MaxQueuesCount> commandPools;
	core::smart_refctd_ptr<nbl::system::ISystem> system;
	core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
	video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
	core::smart_refctd_ptr<nbl::system::ILogger> logger;
	core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;
	video::IGPUObjectFromAssetConverter cpu2gpu;
	core::smart_refctd_ptr<video::IGPUImage> m_swapchainImages[CommonAPI::InitOutput::MaxSwapChainImageCount];

	int32_t m_resourceIx = -1;
	uint32_t m_SwapchainImageIx = ~0u;

	core::smart_refctd_ptr<video::IGPUSemaphore> m_imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> m_renderFinished[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUFence> m_frameComplete[FRAMES_IN_FLIGHT] = { nullptr };

	core::smart_refctd_ptr<video::IGPUCommandBuffer> m_cmdbuf[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUCommandBuffer> m_uploadCmdBuf[FRAMES_IN_FLIGHT] = { nullptr };

	nbl::video::ISwapchain::SCreationParams m_swapchainCreationParams;

	// Related to Drawing Stuff
	Camera2D m_Camera;

	core::smart_refctd_ptr<video::IGPUImageView> pseudoStencilImageView[FRAMES_IN_FLIGHT];
	core::smart_refctd_ptr<video::IGPUBuffer> globalsBuffer[FRAMES_IN_FLIGHT];
	core::smart_refctd_ptr<video::IGPUDescriptorSet> descriptorSets[FRAMES_IN_FLIGHT];
	core::smart_refctd_ptr<video::IGPUGraphicsPipeline> graphicsPipeline;
	core::smart_refctd_ptr<video::IGPUGraphicsPipeline> debugGraphicsPipeline;
	core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> descriptorSetLayout;
	core::smart_refctd_ptr<video::IGPUPipelineLayout> graphicsPipelineLayout;

	core::smart_refctd_ptr<video::IGPUGraphicsPipeline> resolveAlphaGraphicsPipeline;
	core::smart_refctd_ptr<video::IGPUPipelineLayout> resolveAlphaPipeLayout;

	DrawBuffersFiller drawBuffers[FRAMES_IN_FLIGHT];
	CPolyline bigPolyline;
	CPolyline bigPolyline2;

	bool fragmentShaderInterlockEnabled = false;

	// TODO: Needs better info about regular scenes and main limiters to improve the allocations in this function
	void initDrawObjects(uint32_t maxObjects)
	{
		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; ++i)
		{
			drawBuffers[i] = DrawBuffersFiller(core::smart_refctd_ptr(utilities));

			size_t maxIndices = maxObjects * 6u * 2u;
			drawBuffers[i].allocateIndexBuffer(logicalDevice, maxIndices);
			drawBuffers[i].allocateMainObjectsBuffer(logicalDevice, maxObjects);
			drawBuffers[i].allocateDrawObjectsBuffer(logicalDevice, maxObjects * 5u);
			drawBuffers[i].allocateStylesBuffer(logicalDevice, 16u);
			drawBuffers[i].allocateCustomClipProjectionBuffer(logicalDevice, 128u);

			// * 3 because I just assume there is on average 3x beziers per actual object (cause we approximate other curves/arcs with beziers now)
			size_t geometryBufferSize = maxObjects * sizeof(QuadraticBezierInfo) * 3;
			drawBuffers[i].allocateGeometryBuffer(logicalDevice, geometryBufferSize);
		}

		for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i)
		{
			video::IGPUBuffer::SCreationParams globalsCreationParams = {};
			globalsCreationParams.size = sizeof(Globals);
			globalsCreationParams.usage = video::IGPUBuffer::EUF_UNIFORM_BUFFER_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT | video::IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF;
			globalsBuffer[i] = logicalDevice->createBuffer(std::move(globalsCreationParams));

			video::IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = globalsBuffer[i]->getMemoryReqs();
			memReq.memoryTypeBits &= physicalDevice->getDeviceLocalMemoryTypeBits();
			auto globalsBufferMem = logicalDevice->allocate(memReq, globalsBuffer[i].get());
		}

		// pseudoStencil

		asset::E_FORMAT pseudoStencilFormat = asset::EF_R32_UINT;

		video::IPhysicalDevice::SImageFormatPromotionRequest promotionRequest = {};
		promotionRequest.originalFormat = asset::EF_R32_UINT;
		promotionRequest.usages = {};
		promotionRequest.usages.storageImageAtomic = true;
		pseudoStencilFormat = physicalDevice->promoteImageFormat(promotionRequest, video::IGPUImage::ET_OPTIMAL);

		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; ++i)
		{
			video::IGPUImage::SCreationParams imgInfo;
			imgInfo.format = pseudoStencilFormat;
			imgInfo.type = video::IGPUImage::ET_2D;
			imgInfo.extent.width = window->getWidth();
			imgInfo.extent.height = window->getHeight();
			imgInfo.extent.depth = 1u;
			imgInfo.mipLevels = 1u;
			imgInfo.arrayLayers = 1u;
			imgInfo.samples = asset::ICPUImage::ESCF_1_BIT;
			imgInfo.flags = asset::IImage::E_CREATE_FLAGS::ECF_NONE;
			imgInfo.usage = asset::IImage::EUF_STORAGE_BIT | asset::IImage::EUF_TRANSFER_DST_BIT;
			imgInfo.initialLayout = video::IGPUImage::EL_UNDEFINED;
			imgInfo.tiling = video::IGPUImage::ET_OPTIMAL;

			auto image = logicalDevice->createImage(std::move(imgInfo));
			auto imageMemReqs = image->getMemoryReqs();
			imageMemReqs.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
			logicalDevice->allocate(imageMemReqs, image.get());

			image->setObjectDebugName("pseudoStencil Image");

			video::IGPUImageView::SCreationParams imgViewInfo;
			imgViewInfo.image = std::move(image);
			imgViewInfo.format = pseudoStencilFormat;
			imgViewInfo.viewType = video::IGPUImageView::ET_2D;
			imgViewInfo.flags = video::IGPUImageView::E_CREATE_FLAGS::ECF_NONE;
			imgViewInfo.subresourceRange.aspectMask = asset::IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
			imgViewInfo.subresourceRange.baseArrayLayer = 0u;
			imgViewInfo.subresourceRange.baseMipLevel = 0u;
			imgViewInfo.subresourceRange.layerCount = 1u;
			imgViewInfo.subresourceRange.levelCount = 1u;

			pseudoStencilImageView[i] = logicalDevice->createImageView(std::move(imgViewInfo));
		}
	}

public:
	void setWindow(core::smart_refctd_ptr<nbl::ui::IWindow>&& wnd) override
	{
		window = std::move(wnd);
	}
	nbl::ui::IWindow* getWindow() override
	{
		return window.get();
	}
	void setSystem(core::smart_refctd_ptr<nbl::system::ISystem>&& system) override
	{
		system = std::move(system);
	}
	video::IAPIConnection* getAPIConnection() override
	{
		return apiConnection.get();
	}
	video::ILogicalDevice* getLogicalDevice()  override
	{
		return logicalDevice.get();
	}
	video::IGPURenderpass* getRenderpass() override
	{
		return renderpassFinal.get();
	}
	void setSurface(core::smart_refctd_ptr<video::ISurface>&& s) override
	{
		surface = std::move(s);
	}
	void setFBOs(std::vector<core::smart_refctd_ptr<video::IGPUFramebuffer>>& f) override
	{
		for (int i = 0; i < f.size(); i++)
		{
			auto& fboDynArray = *(framebuffersDynArraySmartPtr.get());
			fboDynArray[i] = core::smart_refctd_ptr(f[i]);
		}
	}
	void setSwapchain(core::smart_refctd_ptr<video::ISwapchain>&& s) override
	{
		swapchain = std::move(s);
	}
	uint32_t getSwapchainImageCount() override
	{
		return swapchain->getImageCount();
	}
	virtual nbl::asset::E_FORMAT getDepthFormat() override
	{
		return nbl::asset::EF_UNKNOWN;
	}

	nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> createRenderpass(
		nbl::asset::E_FORMAT colorAttachmentFormat,
		nbl::asset::E_FORMAT baseDepthFormat,
		nbl::video::IGPURenderpass::E_LOAD_OP loadOp,
		nbl::asset::IImage::E_LAYOUT initialLayout,
		nbl::asset::IImage::E_LAYOUT finalLayout)
	{
		using namespace nbl;

		bool useDepth = baseDepthFormat != nbl::asset::EF_UNKNOWN;
		nbl::asset::E_FORMAT depthFormat = nbl::asset::EF_UNKNOWN;
		if (useDepth)
		{
			depthFormat = logicalDevice->getPhysicalDevice()->promoteImageFormat(
				{ baseDepthFormat, nbl::video::IPhysicalDevice::SFormatImageUsages::SUsage(nbl::asset::IImage::EUF_DEPTH_STENCIL_ATTACHMENT_BIT) },
				nbl::video::IGPUImage::ET_OPTIMAL
			);
			assert(depthFormat != nbl::asset::EF_UNKNOWN);
		}

		nbl::video::IGPURenderpass::SCreationParams::SAttachmentDescription attachments[2];
		attachments[0].initialLayout = initialLayout;
		attachments[0].finalLayout = finalLayout;
		attachments[0].format = colorAttachmentFormat;
		attachments[0].samples = asset::IImage::ESCF_1_BIT;
		attachments[0].loadOp = loadOp;
		attachments[0].storeOp = nbl::video::IGPURenderpass::ESO_STORE;

		attachments[1].initialLayout = asset::IImage::EL_UNDEFINED;
		attachments[1].finalLayout = asset::IImage::EL_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		attachments[1].format = depthFormat;
		attachments[1].samples = asset::IImage::ESCF_1_BIT;
		attachments[1].loadOp = loadOp;
		attachments[1].storeOp = nbl::video::IGPURenderpass::ESO_STORE;

		nbl::video::IGPURenderpass::SCreationParams::SSubpassDescription::SAttachmentRef colorAttRef;
		colorAttRef.attachment = 0u;
		colorAttRef.layout = asset::IImage::EL_COLOR_ATTACHMENT_OPTIMAL;

		nbl::video::IGPURenderpass::SCreationParams::SSubpassDescription::SAttachmentRef depthStencilAttRef;
		depthStencilAttRef.attachment = 1u;
		depthStencilAttRef.layout = asset::IImage::EL_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		nbl::video::IGPURenderpass::SCreationParams::SSubpassDescription sp;
		sp.pipelineBindPoint = asset::EPBP_GRAPHICS;
		sp.colorAttachmentCount = 1u;
		sp.colorAttachments = &colorAttRef;
		if (useDepth) {
			sp.depthStencilAttachment = &depthStencilAttRef;
		}
		else {
			sp.depthStencilAttachment = nullptr;
		}
		sp.flags = nbl::video::IGPURenderpass::ESDF_NONE;
		sp.inputAttachmentCount = 0u;
		sp.inputAttachments = nullptr;
		sp.preserveAttachmentCount = 0u;
		sp.preserveAttachments = nullptr;
		sp.resolveAttachments = nullptr;

		nbl::video::IGPURenderpass::SCreationParams rp_params;
		rp_params.attachmentCount = (useDepth) ? 2u : 1u;
		rp_params.attachments = attachments;
		rp_params.dependencies = nullptr;
		rp_params.dependencyCount = 0u;
		rp_params.subpasses = &sp;
		rp_params.subpassCount = 1u;

		return logicalDevice->createRenderpass(rp_params);
	}

	void getAndLogQueryPoolResults()
	{
#ifdef BEZIER_CAGE_ADAPTIVE_T_FIND // results for bezier show an optimal number of 0.14 for T
		{
			uint32_t samples_passed[1] = {};
			auto queryResultFlags = core::bitflag<video::IQueryPool::E_QUERY_RESULTS_FLAGS>(video::IQueryPool::EQRF_WAIT_BIT);
			logicalDevice->getQueryPoolResults(pipelineStatsPool.get(), 0u, 1u, sizeof(samples_passed), samples_passed, sizeof(uint32_t), queryResultFlags);
			logger->log("[WAIT] SamplesPassed[0] = %d", system::ILogger::ELL_INFO, samples_passed[0]);
			std::cout << MinT << ", " << PrevSamples << std::endl;
			if (PrevSamples > samples_passed[0]) {
				PrevSamples = samples_passed[0];
				MinT = (sin(T) + 1.01f) / 4.03f;
			}
		}
#endif
	}

	APP_CONSTRUCTOR(CADApp);

	void onAppInitialized_impl() override
	{
		std::this_thread::sleep_for(std::chrono::seconds(5));

		const auto swapchainImageUsage = static_cast<asset::IImage::E_USAGE_FLAGS>(asset::IImage::EUF_COLOR_ATTACHMENT_BIT);
		std::array<asset::E_FORMAT, 1> acceptableSurfaceFormats = { asset::EF_B8G8R8A8_UNORM };

		CommonAPI::InitParams initParams;
		initParams.windowCb = core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback>(this);
		initParams.window = core::smart_refctd_ptr(window);
		initParams.apiType = video::EAT_VULKAN;
		initParams.appName = { "62.CAD" };
		initParams.framesInFlight = FRAMES_IN_FLIGHT;
		initParams.windowWidth = REQUESTED_WIN_W;
		initParams.windowHeight = REQUESTED_WIN_H;
		initParams.swapchainImageCount = 3u;
		initParams.swapchainImageUsage = swapchainImageUsage;
		initParams.depthFormat = getDepthFormat();
		initParams.acceptableSurfaceFormats = acceptableSurfaceFormats.data();
		initParams.acceptableSurfaceFormatCount = acceptableSurfaceFormats.size();
		initParams.physicalDeviceFilter.requiredFeatures.bufferDeviceAddress = true;
		initParams.physicalDeviceFilter.requiredFeatures.shaderFloat64 = true;
		initParams.physicalDeviceFilter.requiredFeatures.fillModeNonSolid = DebugMode;
		initParams.physicalDeviceFilter.requiredFeatures.fragmentShaderPixelInterlock = FragmentShaderPixelInterlock;
		initParams.physicalDeviceFilter.requiredFeatures.pipelineStatisticsQuery = true;
		initParams.physicalDeviceFilter.requiredFeatures.shaderClipDistance = true;
		initParams.physicalDeviceFilter.requiredFeatures.scalarBlockLayout = true;
		auto initOutput = CommonAPI::InitWithDefaultExt(std::move(initParams));

		system = std::move(initOutput.system);
		window = std::move(initParams.window);
		windowCb = std::move(initParams.windowCb);
		apiConnection = std::move(initOutput.apiConnection);
		surface = std::move(initOutput.surface);
		physicalDevice = std::move(initOutput.physicalDevice);
		logicalDevice = std::move(initOutput.logicalDevice);
		utilities = std::move(initOutput.utilities);
		queues = std::move(initOutput.queues);
		assetManager = std::move(initOutput.assetManager);
		cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
		logger = std::move(initOutput.logger);
		inputSystem = std::move(initOutput.inputSystem);
		windowManager = std::move(initOutput.windowManager);
		// renderpass = std::move(initOutput.renderToSwapchainRenderpass);
		m_swapchainCreationParams = std::move(initOutput.swapchainCreationParams);

		fragmentShaderInterlockEnabled = logicalDevice->getEnabledFeatures().fragmentShaderPixelInterlock;

		{
			video::IQueryPool::SCreationParams queryPoolCreationParams = {};
			queryPoolCreationParams.queryType = video::IQueryPool::EQT_PIPELINE_STATISTICS;
			queryPoolCreationParams.queryCount = 1u;
			queryPoolCreationParams.pipelineStatisticsFlags = video::IQueryPool::EPSF_FRAGMENT_SHADER_INVOCATIONS_BIT;
			pipelineStatsPool = logicalDevice->createQueryPool(std::move(queryPoolCreationParams));
		}

		logger->log("dupa", system::ILogger::ELL_INFO);

		renderpassInitial = createRenderpass(m_swapchainCreationParams.surfaceFormat.format, getDepthFormat(), nbl::video::IGPURenderpass::ELO_CLEAR, asset::IImage::EL_UNDEFINED, asset::IImage::EL_COLOR_ATTACHMENT_OPTIMAL);
		renderpassInBetween = createRenderpass(m_swapchainCreationParams.surfaceFormat.format, getDepthFormat(), nbl::video::IGPURenderpass::ELO_LOAD, asset::IImage::EL_COLOR_ATTACHMENT_OPTIMAL, asset::IImage::EL_COLOR_ATTACHMENT_OPTIMAL);
		renderpassFinal = createRenderpass(m_swapchainCreationParams.surfaceFormat.format, getDepthFormat(), nbl::video::IGPURenderpass::ELO_LOAD, asset::IImage::EL_COLOR_ATTACHMENT_OPTIMAL, asset::IImage::EL_PRESENT_SRC);

		commandPools = std::move(initOutput.commandPools);
		const auto& graphicsCommandPools = commandPools[CommonAPI::InitOutput::EQT_GRAPHICS];
		const auto& transferCommandPools = commandPools[CommonAPI::InitOutput::EQT_TRANSFER_UP];

		CommonAPI::createSwapchain(std::move(logicalDevice), m_swapchainCreationParams, window->getWidth(), window->getHeight(), swapchain);

		framebuffersDynArraySmartPtr = CommonAPI::createFBOWithSwapchainImages(
			swapchain->getImageCount(), window->getWidth(), window->getHeight(),
			logicalDevice, swapchain, renderpassFinal,
			getDepthFormat()
		);

		const uint32_t swapchainImageCount = swapchain->getImageCount();
		for (uint32_t i = 0; i < swapchainImageCount; ++i)
		{
			auto& fboDynArray = *(framebuffersDynArraySmartPtr.get());
			m_swapchainImages[i] = fboDynArray[i]->getCreationParameters().attachments[0u]->getCreationParameters().image;
		}

		video::IGPUObjectFromAssetConverter CPU2GPU;

		core::smart_refctd_ptr<video::IGPUSpecializedShader> shaders[4u] = {};
		{
			asset::IAssetLoader::SAssetLoadParams params = {};
			params.logger = logger.get();
			core::smart_refctd_ptr<asset::ICPUSpecializedShader> cpuShaders[4u] = {};
			constexpr auto vertexShaderPath = "../vertex_shader.hlsl";
			constexpr auto fragmentShaderPath = "../fragment_shader.hlsl";
			constexpr auto debugfragmentShaderPath = "../fragment_shader_debug.hlsl";
			constexpr auto resolveAlphasShaderPath = "../resolve_alphas.hlsl";
			cpuShaders[0u] = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*assetManager->getAsset(vertexShaderPath, params).getContents().begin());
			cpuShaders[1u] = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*assetManager->getAsset(fragmentShaderPath, params).getContents().begin());
			cpuShaders[2u] = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*assetManager->getAsset(debugfragmentShaderPath, params).getContents().begin());
			cpuShaders[3u] = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*assetManager->getAsset(resolveAlphasShaderPath, params).getContents().begin());
			cpuShaders[0u]->setSpecializationInfo(asset::ISpecializedShader::SInfo(nullptr, nullptr, "main"));
			cpuShaders[1u]->setSpecializationInfo(asset::ISpecializedShader::SInfo(nullptr, nullptr, "main"));
			cpuShaders[2u]->setSpecializationInfo(asset::ISpecializedShader::SInfo(nullptr, nullptr, "main"));
			cpuShaders[3u]->setSpecializationInfo(asset::ISpecializedShader::SInfo(nullptr, nullptr, "main"));
			auto gpuShaders = CPU2GPU.getGPUObjectsFromAssets(cpuShaders, cpuShaders + 4u, cpu2gpuParams);
			shaders[0u] = gpuShaders->begin()[0u];
			shaders[1u] = gpuShaders->begin()[1u];
			shaders[2u] = gpuShaders->begin()[2u];
			shaders[3u] = gpuShaders->begin()[3u];
		}

		initDrawObjects(20480u);

		// Create DescriptorSetLayout, PipelineLayout and update DescriptorSets
		{
			constexpr uint32_t BindingCount = 6u;
			video::IGPUDescriptorSetLayout::SBinding bindings[BindingCount] = {};
			bindings[0].binding = 0u;
			bindings[0].type = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER;
			bindings[0].count = 1u;
			bindings[0].stageFlags = asset::IShader::ESS_VERTEX | asset::IShader::ESS_FRAGMENT;

			bindings[1].binding = 1u;
			bindings[1].type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
			bindings[1].count = 1u;
			bindings[1].stageFlags = asset::IShader::ESS_VERTEX | asset::IShader::ESS_FRAGMENT;

			bindings[2].binding = 2u;
			bindings[2].type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE;
			bindings[2].count = 1u;
			bindings[2].stageFlags = asset::IShader::ESS_FRAGMENT;

			bindings[3].binding = 3u;
			bindings[3].type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
			bindings[3].count = 1u;
			bindings[3].stageFlags = asset::IShader::ESS_VERTEX | asset::IShader::ESS_FRAGMENT;

			bindings[4].binding = 4u;
			bindings[4].type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
			bindings[4].count = 1u;
			bindings[4].stageFlags = asset::IShader::ESS_VERTEX | asset::IShader::ESS_FRAGMENT;

			bindings[5].binding = 5u;
			bindings[5].type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
			bindings[5].count = 1u;
			bindings[5].stageFlags = asset::IShader::ESS_VERTEX;

			descriptorSetLayout = logicalDevice->createDescriptorSetLayout(bindings, bindings + BindingCount);

			nbl::core::smart_refctd_ptr<nbl::video::IDescriptorPool> descriptorPool = nullptr;
			{
				nbl::video::IDescriptorPool::SCreateInfo createInfo = {};
				createInfo.flags = nbl::video::IDescriptorPool::ECF_NONE;
				createInfo.maxSets = 128u;
				createInfo.maxDescriptorCount[static_cast<uint32_t>(nbl::asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER)] = FRAMES_IN_FLIGHT;
				createInfo.maxDescriptorCount[static_cast<uint32_t>(nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER)] = 4 * FRAMES_IN_FLIGHT;
				createInfo.maxDescriptorCount[static_cast<uint32_t>(nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE)] = FRAMES_IN_FLIGHT;

				descriptorPool = logicalDevice->createDescriptorPool(std::move(createInfo));
			}

			for (size_t i = 0; i < FRAMES_IN_FLIGHT; i++)
			{
				descriptorSets[i] = descriptorPool->createDescriptorSet(core::smart_refctd_ptr(descriptorSetLayout));
				video::IGPUDescriptorSet::SDescriptorInfo descriptorInfos[6u] = {};
				descriptorInfos[0u].info.buffer.offset = 0u;
				descriptorInfos[0u].info.buffer.size = globalsBuffer[i]->getCreationParams().size;
				descriptorInfos[0u].desc = globalsBuffer[i];

				descriptorInfos[1u].info.buffer.offset = 0u;
				descriptorInfos[1u].info.buffer.size = drawBuffers[i].gpuDrawBuffers.drawObjectsBuffer->getCreationParams().size;
				descriptorInfos[1u].desc = drawBuffers[i].gpuDrawBuffers.drawObjectsBuffer;

				descriptorInfos[2u].info.image.imageLayout = asset::IImage::E_LAYOUT::EL_GENERAL;
				descriptorInfos[2u].info.image.sampler = nullptr;
				descriptorInfos[2u].desc = pseudoStencilImageView[i];

				descriptorInfos[3u].info.buffer.offset = 0u;
				descriptorInfos[3u].info.buffer.size = drawBuffers[i].gpuDrawBuffers.lineStylesBuffer->getCreationParams().size;
				descriptorInfos[3u].desc = drawBuffers[i].gpuDrawBuffers.lineStylesBuffer;

				descriptorInfos[4u].info.buffer.offset = 0u;
				descriptorInfos[4u].info.buffer.size = drawBuffers[i].gpuDrawBuffers.mainObjectsBuffer->getCreationParams().size;
				descriptorInfos[4u].desc = drawBuffers[i].gpuDrawBuffers.mainObjectsBuffer;

				descriptorInfos[5u].info.buffer.offset = 0u;
				descriptorInfos[5u].info.buffer.size = drawBuffers[i].gpuDrawBuffers.customClipProjectionBuffer->getCreationParams().size;
				descriptorInfos[5u].desc = drawBuffers[i].gpuDrawBuffers.customClipProjectionBuffer;

				video::IGPUDescriptorSet::SWriteDescriptorSet descriptorUpdates[6u] = {};
				descriptorUpdates[0u].dstSet = descriptorSets[i].get();
				descriptorUpdates[0u].binding = 0u;
				descriptorUpdates[0u].arrayElement = 0u;
				descriptorUpdates[0u].count = 1u;
				descriptorUpdates[0u].descriptorType = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER;
				descriptorUpdates[0u].info = &descriptorInfos[0u];

				descriptorUpdates[1u].dstSet = descriptorSets[i].get();
				descriptorUpdates[1u].binding = 1u;
				descriptorUpdates[1u].arrayElement = 0u;
				descriptorUpdates[1u].count = 1u;
				descriptorUpdates[1u].descriptorType = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
				descriptorUpdates[1u].info = &descriptorInfos[1u];

				descriptorUpdates[2u].dstSet = descriptorSets[i].get();
				descriptorUpdates[2u].binding = 2u;
				descriptorUpdates[2u].arrayElement = 0u;
				descriptorUpdates[2u].count = 1u;
				descriptorUpdates[2u].descriptorType = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE;
				descriptorUpdates[2u].info = &descriptorInfos[2u];

				descriptorUpdates[3u].dstSet = descriptorSets[i].get();
				descriptorUpdates[3u].binding = 3u;
				descriptorUpdates[3u].arrayElement = 0u;
				descriptorUpdates[3u].count = 1u;
				descriptorUpdates[3u].descriptorType = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
				descriptorUpdates[3u].info = &descriptorInfos[3u];

				descriptorUpdates[4u].dstSet = descriptorSets[i].get();
				descriptorUpdates[4u].binding = 4u;
				descriptorUpdates[4u].arrayElement = 0u;
				descriptorUpdates[4u].count = 1u;
				descriptorUpdates[4u].descriptorType = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
				descriptorUpdates[4u].info = &descriptorInfos[4u];

				descriptorUpdates[5u].dstSet = descriptorSets[i].get();
				descriptorUpdates[5u].binding = 5u;
				descriptorUpdates[5u].arrayElement = 0u;
				descriptorUpdates[5u].count = 1u;
				descriptorUpdates[5u].descriptorType = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
				descriptorUpdates[5u].info = &descriptorInfos[5u];

				logicalDevice->updateDescriptorSets(6u, descriptorUpdates, 0u, nullptr);
			}

			graphicsPipelineLayout = logicalDevice->createPipelineLayout(nullptr, nullptr, core::smart_refctd_ptr(descriptorSetLayout), nullptr, nullptr, nullptr);
		}

		// Shared Blend Params between pipelines
		asset::SBlendParams blendParams = {};
		blendParams.blendParams[0u].blendEnable = true;
		blendParams.blendParams[0u].srcColorFactor = asset::EBF_SRC_ALPHA;
		blendParams.blendParams[0u].dstColorFactor = asset::EBF_ONE_MINUS_SRC_ALPHA;
		blendParams.blendParams[0u].colorBlendOp = asset::EBO_ADD;
		blendParams.blendParams[0u].srcAlphaFactor = asset::EBF_ONE;
		blendParams.blendParams[0u].dstAlphaFactor = asset::EBF_ZERO;
		blendParams.blendParams[0u].alphaBlendOp = asset::EBO_ADD;
		blendParams.blendParams[0u].colorWriteMask = (1u << 4u) - 1u;

		// Create Alpha Resovle Pipeline
		{
			auto fsTriangleProtoPipe = nbl::ext::FullScreenTriangle::createProtoPipeline(cpu2gpuParams, 0u);
			std::get<asset::SBlendParams>(fsTriangleProtoPipe) = blendParams;

			auto constants = std::get<asset::SPushConstantRange>(fsTriangleProtoPipe);
			resolveAlphaPipeLayout = logicalDevice->createPipelineLayout(&constants, &constants+1, core::smart_refctd_ptr(descriptorSetLayout), nullptr, nullptr, nullptr);
			auto fsTriangleRenderPassIndependantPipe = nbl::ext::FullScreenTriangle::createRenderpassIndependentPipeline(logicalDevice.get(), fsTriangleProtoPipe, core::smart_refctd_ptr(shaders[3u]), core::smart_refctd_ptr(resolveAlphaPipeLayout));

			video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineCreateInfo = {};
			graphicsPipelineCreateInfo.renderpassIndependent = fsTriangleRenderPassIndependantPipe;
			graphicsPipelineCreateInfo.renderpass = renderpassFinal;
			resolveAlphaGraphicsPipeline = logicalDevice->createGraphicsPipeline(nullptr, std::move(graphicsPipelineCreateInfo));
		}

		// Create Main Graphics Pipelines 
		{
			video::IGPURenderpassIndependentPipeline::SCreationParams renderpassIndependantPipeInfo = {};
			renderpassIndependantPipeInfo.layout = graphicsPipelineLayout;
			renderpassIndependantPipeInfo.shaders[0u] = shaders[0u];
			renderpassIndependantPipeInfo.shaders[1u] = shaders[1u];
			// renderpassIndependantPipeInfo.vertexInput; no gpu vertex buffers
			renderpassIndependantPipeInfo.blend = blendParams;

			renderpassIndependantPipeInfo.primitiveAssembly.primitiveType = asset::E_PRIMITIVE_TOPOLOGY::EPT_TRIANGLE_LIST;
			renderpassIndependantPipeInfo.rasterization.depthTestEnable = false;
			renderpassIndependantPipeInfo.rasterization.depthWriteEnable = false;
			renderpassIndependantPipeInfo.rasterization.stencilTestEnable = false;
			renderpassIndependantPipeInfo.rasterization.polygonMode = asset::EPM_FILL;
			renderpassIndependantPipeInfo.rasterization.faceCullingMode = asset::EFCM_NONE;

			core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> renderpassIndependant;
			bool succ = logicalDevice->createRenderpassIndependentPipelines(
				nullptr,
				core::SRange<const video::IGPURenderpassIndependentPipeline::SCreationParams>(&renderpassIndependantPipeInfo, &renderpassIndependantPipeInfo + 1u),
				&renderpassIndependant);
			assert(succ);

			video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineCreateInfo = {};
			graphicsPipelineCreateInfo.renderpassIndependent = renderpassIndependant;
			graphicsPipelineCreateInfo.renderpass = renderpassFinal;
			graphicsPipeline = logicalDevice->createGraphicsPipeline(nullptr, std::move(graphicsPipelineCreateInfo));

			if constexpr (DebugMode)
			{
				core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> renderpassIndependantDebug;
				renderpassIndependantPipeInfo.shaders[1u] = shaders[2u];
				renderpassIndependantPipeInfo.rasterization.polygonMode = asset::EPM_LINE;
				succ = logicalDevice->createRenderpassIndependentPipelines(
					nullptr,
					core::SRange<const video::IGPURenderpassIndependentPipeline::SCreationParams>(&renderpassIndependantPipeInfo, &renderpassIndependantPipeInfo + 1u),
					&renderpassIndependantDebug);
				assert(succ);

				video::IGPUGraphicsPipeline::SCreationParams debugGraphicsPipelineCreateInfo = {};
				debugGraphicsPipelineCreateInfo.renderpassIndependent = renderpassIndependantDebug;
				debugGraphicsPipelineCreateInfo.renderpass = renderpassFinal;
				debugGraphicsPipeline = logicalDevice->createGraphicsPipeline(nullptr, std::move(debugGraphicsPipelineCreateInfo));
			}
		}

		for (size_t i = 0; i < FRAMES_IN_FLIGHT; i++)
		{
			logicalDevice->createCommandBuffers(
				graphicsCommandPools[i].get(),
				video::IGPUCommandBuffer::EL_PRIMARY,
				1,
				m_cmdbuf + i);

			logicalDevice->createCommandBuffers(
				transferCommandPools[i].get(),
				video::IGPUCommandBuffer::EL_PRIMARY,
				1,
				m_uploadCmdBuf + i);
		}

		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
		{
			m_frameComplete[i] = logicalDevice->createFence(video::IGPUFence::ECF_SIGNALED_BIT);
			m_imageAcquire[i] = logicalDevice->createSemaphore();
			m_renderFinished[i] = logicalDevice->createSemaphore();
		}

		m_Camera.setOrigin({ 0.0, 0.0 });
		m_Camera.setAspectRatio((double)window->getWidth() / window->getHeight());
		m_Camera.setSize(200.0);

		m_timeElapsed = 0.0;


		if constexpr (mode == ExampleMode::CASE_1)
		{
			{
				std::vector<float64_t2> linePoints;
				for (uint32_t i = 0u; i < 20u; ++i)
				{
					for (uint32_t i = 0u; i < 256u; ++i)
					{
						double y = -112.0 + i * 1.1;
						linePoints.push_back({ -200.0, y });
						linePoints.push_back({ +200.0, y });
					}
					for (uint32_t i = 0u; i < 256u; ++i)
					{
						double x = -200.0 + i * 1.5;
						linePoints.push_back({ x, -100.0 });
						linePoints.push_back({ x, +100.0 });
					}
				}
				bigPolyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
			}
			{
				std::vector<float64_t2> linePoints;
				for (uint32_t i = 0u; i < 20u; ++i)
				{
					for (uint32_t i = 0u; i < 256u; ++i)
					{
						double y = -112.0 + i * 1.1;
						double x = -200.0 + i * 1.5;
						linePoints.push_back({ -200.0 + x, y });
						linePoints.push_back({ +200.0 + x, y });
					}
					for (uint32_t i = 0u; i < 256u; ++i)
					{
						double y = -112.0 + i * 1.1;
						double x = -200.0 + i * 1.5;
						linePoints.push_back({ x, -100.0 + y });
						linePoints.push_back({ x, +100.0 + y });
					}
				}
				bigPolyline2.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
			}
		}

	}

	void onAppTerminated_impl() override
	{
		logicalDevice->waitIdle();
	}

	float getScreenToWorldRatio(const float64_t3x3& viewProjectionMatrix, uint32_t2 windowSize)
	{
		double idx_0_0 = viewProjectionMatrix[0u][0u] * (windowSize.x / 2.0);
		double idx_1_1 = viewProjectionMatrix[1u][1u] * (windowSize.y / 2.0);
		double det_2x2_mat = idx_0_0 * idx_1_1;
		return static_cast<float>(core::sqrt(core::abs(det_2x2_mat)));
	}

	void beginFrameRender()
	{
		auto& cb = m_cmdbuf[m_resourceIx];
		auto& commandPool = commandPools[CommonAPI::InitOutput::EQT_GRAPHICS][m_resourceIx];
		auto& fence = m_frameComplete[m_resourceIx];
		logicalDevice->blockForFences(1u, &fence.get());
		logicalDevice->resetFences(1u, &fence.get());

		m_SwapchainImageIx = 0u;
		auto acquireResult = swapchain->acquireNextImage(m_imageAcquire[m_resourceIx].get(), nullptr, &m_SwapchainImageIx);
		assert(acquireResult == video::ISwapchain::E_ACQUIRE_IMAGE_RESULT::EAIR_SUCCESS);

		core::smart_refctd_ptr<video::IGPUImage> swapchainImg = m_swapchainImages[m_SwapchainImageIx];

		// safe to proceed
		cb->reset(video::IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT); // TODO: Begin doesn't release the resources in the command pool, meaning the old swapchains never get dropped
		cb->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT); // TODO: Reset Frame's CommandPool
		cb->beginDebugMarker("Frame");
		Globals globalData = {};
		globalData.antiAliasingFactor = 1.0f;// + abs(cos(m_timeElapsed * 0.0008))*20.0f;
		globalData.resolution = uint32_t2{ WIN_W, WIN_H };
		globalData.defaultClipProjection.projectionToNDC = m_Camera.constructViewProjection(m_timeElapsed);
		globalData.defaultClipProjection.minClipNDC = float32_t2(-1.0, -1.0);
		globalData.defaultClipProjection.maxClipNDC = float32_t2(+1.0, +1.0);
		globalData.screenToWorldRatio = getScreenToWorldRatio(globalData.defaultClipProjection.projectionToNDC, globalData.resolution);
		globalData.worldToScreenRatio = 1.0f/globalData.screenToWorldRatio;
		globalData.majorAxis = HatchMajorAxis;
		bool updateSuccess = cb->updateBuffer(globalsBuffer[m_resourceIx].get(), 0ull, sizeof(Globals), &globalData);
		assert(updateSuccess);

		// Clear pseudoStencil
		{
			auto pseudoStencilImage = pseudoStencilImageView[m_resourceIx]->getCreationParameters().image;

			nbl::video::IGPUCommandBuffer::SImageMemoryBarrier imageBarriers[1u] = {};
			imageBarriers[0].barrier.srcAccessMask = nbl::asset::EAF_NONE;
			imageBarriers[0].barrier.dstAccessMask = nbl::asset::EAF_MEMORY_WRITE_BIT;
			imageBarriers[0].oldLayout = nbl::asset::IImage::EL_UNDEFINED;
			imageBarriers[0].newLayout = nbl::asset::IImage::EL_GENERAL;
			imageBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageBarriers[0].image = pseudoStencilImage;
			imageBarriers[0].subresourceRange.aspectMask = nbl::asset::IImage::EAF_COLOR_BIT;
			imageBarriers[0].subresourceRange.baseMipLevel = 0u;
			imageBarriers[0].subresourceRange.levelCount = 1;
			imageBarriers[0].subresourceRange.baseArrayLayer = 0u;
			imageBarriers[0].subresourceRange.layerCount = 1;
			cb->pipelineBarrier(nbl::asset::EPSF_TOP_OF_PIPE_BIT, nbl::asset::EPSF_TRANSFER_BIT, nbl::asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, imageBarriers);

			uint32_t pseudoStencilInvalidValue = core::bitfieldInsert<uint32_t>(0u, InvalidMainObjectIdx, AlphaBits, MainObjectIdxBits);
			asset::SClearColorValue clear = {};
			clear.uint32[0] = pseudoStencilInvalidValue;

			asset::IImage::SSubresourceRange subresourceRange = {};
			subresourceRange.aspectMask = asset::IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
			subresourceRange.baseArrayLayer = 0u;
			subresourceRange.baseMipLevel = 0u;
			subresourceRange.layerCount = 1u;
			subresourceRange.levelCount = 1u;

			cb->clearColorImage(pseudoStencilImage.get(), asset::IImage::EL_GENERAL, &clear, 1u, &subresourceRange);
		}

		nbl::video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
		{
			VkRect2D area;
			area.offset = { 0,0 };
			area.extent = { window->getWidth(), window->getHeight() };
			asset::SClearValue clear[2] = {};
			clear[0].color.float32[0] = 0.8f;
			clear[0].color.float32[1] = 0.8f;
			clear[0].color.float32[2] = 0.8f;
			clear[0].color.float32[3] = 0.f;
			clear[1].depthStencil.depth = 1.f;

			beginInfo.clearValueCount = 2u;
			beginInfo.framebuffer = framebuffersDynArraySmartPtr->begin()[m_SwapchainImageIx];
			beginInfo.renderpass = renderpassInitial;
			beginInfo.renderArea = area;
			beginInfo.clearValues = clear;
		}

		// you could do this later but only use renderpassInitial on first draw
		cb->beginRenderPass(&beginInfo, asset::ESC_INLINE);
		cb->endRenderPass();
	}

	void pipelineBarriersBeforeDraw(video::IGPUCommandBuffer* const cb)
	{
		auto& currentDrawBuffers = drawBuffers[m_resourceIx];
		{
			auto pseudoStencilImage = pseudoStencilImageView[m_resourceIx]->getCreationParameters().image;
			nbl::video::IGPUCommandBuffer::SImageMemoryBarrier imageBarriers[1u] = {};
			imageBarriers[0].barrier.srcAccessMask = nbl::asset::EAF_MEMORY_WRITE_BIT;
			imageBarriers[0].barrier.dstAccessMask = nbl::asset::EAF_SHADER_READ_BIT | nbl::asset::EAF_SHADER_WRITE_BIT; // SYNC_FRAGMENT_SHADER_SHADER_SAMPLED_READ | SYNC_FRAGMENT_SHADER_SHADER_STORAGE_READ | SYNC_FRAGMENT_SHADER_UNIFORM_READ
			imageBarriers[0].oldLayout = nbl::asset::IImage::EL_GENERAL;
			imageBarriers[0].newLayout = nbl::asset::IImage::EL_GENERAL;
			imageBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageBarriers[0].image = pseudoStencilImage;
			imageBarriers[0].subresourceRange.aspectMask = nbl::asset::IImage::EAF_COLOR_BIT;
			imageBarriers[0].subresourceRange.baseMipLevel = 0u;
			imageBarriers[0].subresourceRange.levelCount = 1;
			imageBarriers[0].subresourceRange.baseArrayLayer = 0u;
			imageBarriers[0].subresourceRange.layerCount = 1;
			cb->pipelineBarrier(nbl::asset::EPSF_TRANSFER_BIT, nbl::asset::EPSF_FRAGMENT_SHADER_BIT, nbl::asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, imageBarriers);

		}
		{
			nbl::video::IGPUCommandBuffer::SBufferMemoryBarrier bufferBarriers[1u] = {};
			bufferBarriers[0u].barrier.srcAccessMask = nbl::asset::EAF_MEMORY_WRITE_BIT;
			bufferBarriers[0u].barrier.dstAccessMask = nbl::asset::EAF_INDEX_READ_BIT;
			bufferBarriers[0u].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			bufferBarriers[0u].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			bufferBarriers[0u].buffer = currentDrawBuffers.gpuDrawBuffers.indexBuffer;
			bufferBarriers[0u].offset = 0u;
			bufferBarriers[0u].size = currentDrawBuffers.getCurrentIndexBufferSize();
			cb->pipelineBarrier(nbl::asset::EPSF_TRANSFER_BIT, nbl::asset::EPSF_VERTEX_INPUT_BIT, nbl::asset::EDF_NONE, 0u, nullptr, 1u, bufferBarriers, 0u, nullptr);
		}
		{
			constexpr uint32_t MaxBufferBarriersCount = 5u;
			uint32_t bufferBarriersCount = 0u;
			nbl::video::IGPUCommandBuffer::SBufferMemoryBarrier bufferBarriers[MaxBufferBarriersCount] = {};

			if (globalsBuffer[m_resourceIx]->getSize() > 0u)
			{
				auto& bufferBarrier = bufferBarriers[bufferBarriersCount++];
				bufferBarrier.barrier.srcAccessMask = nbl::asset::EAF_MEMORY_WRITE_BIT;
				bufferBarrier.barrier.dstAccessMask = nbl::asset::EAF_UNIFORM_READ_BIT;
				bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				bufferBarrier.buffer = globalsBuffer[m_resourceIx];
				bufferBarrier.offset = 0u;
				bufferBarrier.size = globalsBuffer[m_resourceIx]->getSize();
			}
			if (currentDrawBuffers.getCurrentDrawObjectsBufferSize() > 0u)
			{
				auto& bufferBarrier = bufferBarriers[bufferBarriersCount++];
				bufferBarrier.barrier.srcAccessMask = nbl::asset::EAF_MEMORY_WRITE_BIT;
				bufferBarrier.barrier.dstAccessMask = nbl::asset::EAF_SHADER_READ_BIT;
				bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				bufferBarrier.buffer = currentDrawBuffers.gpuDrawBuffers.drawObjectsBuffer;
				bufferBarrier.offset = 0u;
				bufferBarrier.size = currentDrawBuffers.getCurrentDrawObjectsBufferSize();
			}
			if (currentDrawBuffers.getCurrentGeometryBufferSize() > 0u)
			{
				auto& bufferBarrier = bufferBarriers[bufferBarriersCount++];
				bufferBarrier.barrier.srcAccessMask = nbl::asset::EAF_MEMORY_WRITE_BIT;
				bufferBarrier.barrier.dstAccessMask = nbl::asset::EAF_SHADER_READ_BIT;
				bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				bufferBarrier.buffer = currentDrawBuffers.gpuDrawBuffers.geometryBuffer;
				bufferBarrier.offset = 0u;
				bufferBarrier.size = currentDrawBuffers.getCurrentGeometryBufferSize();
			}
			if (currentDrawBuffers.getCurrentLineStylesBufferSize() > 0u)
			{
				auto& bufferBarrier = bufferBarriers[bufferBarriersCount++];
				bufferBarrier.barrier.srcAccessMask = nbl::asset::EAF_MEMORY_WRITE_BIT;
				bufferBarrier.barrier.dstAccessMask = nbl::asset::EAF_SHADER_READ_BIT;
				bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				bufferBarrier.buffer = currentDrawBuffers.gpuDrawBuffers.lineStylesBuffer;
				bufferBarrier.offset = 0u;
				bufferBarrier.size = currentDrawBuffers.getCurrentLineStylesBufferSize();
			}
			if (currentDrawBuffers.getCurrentCustomClipProjectionBufferSize() > 0u)
			{
				auto& bufferBarrier = bufferBarriers[bufferBarriersCount++];
				bufferBarrier.barrier.srcAccessMask = nbl::asset::EAF_MEMORY_WRITE_BIT;
				bufferBarrier.barrier.dstAccessMask = nbl::asset::EAF_SHADER_READ_BIT;
				bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				bufferBarrier.buffer = currentDrawBuffers.gpuDrawBuffers.customClipProjectionBuffer;
				bufferBarrier.offset = 0u;
				bufferBarrier.size = currentDrawBuffers.getCurrentCustomClipProjectionBufferSize();
			}
			cb->pipelineBarrier(nbl::asset::EPSF_TRANSFER_BIT, nbl::asset::EPSF_VERTEX_SHADER_BIT | nbl::asset::EPSF_FRAGMENT_SHADER_BIT, nbl::asset::EDF_NONE, 0u, nullptr, bufferBarriersCount, bufferBarriers, 0u, nullptr);
		}
	}

	void endFrameRender()
	{
		auto& cb = m_cmdbuf[m_resourceIx];

		uint32_t windowWidth = swapchain->getCreationParameters().width;
		uint32_t windowHeight = swapchain->getCreationParameters().height;

		asset::SViewport vp;
		vp.minDepth = 1.f;
		vp.maxDepth = 0.f;
		vp.x = 0u;
		vp.y = 0u;
		vp.width = windowWidth;
		vp.height = windowHeight;
		cb->setViewport(0u, 1u, &vp);

		VkRect2D scissor;
		scissor.extent = { windowWidth, windowHeight };
		scissor.offset = { 0, 0 };
		cb->setScissor(0u, 1u, &scissor);

		nbl::video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
		{
			VkRect2D area;
			area.offset = { 0,0 };
			area.extent = { window->getWidth(), window->getHeight() };

			beginInfo.clearValueCount = 0u;
			beginInfo.framebuffer = framebuffersDynArraySmartPtr->begin()[m_SwapchainImageIx];
			beginInfo.renderpass = renderpassFinal;
			beginInfo.renderArea = area;
			beginInfo.clearValues = nullptr;
		}

		pipelineBarriersBeforeDraw(cb.get());

		cb->resetQueryPool(pipelineStatsPool.get(), 0u, 1u);
		cb->beginQuery(pipelineStatsPool.get(), 0);

		cb->beginRenderPass(&beginInfo, asset::ESC_INLINE);

		const uint32_t currentIndexCount = drawBuffers[m_resourceIx].getIndexCount();
		cb->bindDescriptorSets(asset::EPBP_GRAPHICS, graphicsPipelineLayout.get(), 0u, 1u, &descriptorSets[m_resourceIx].get());
		cb->bindIndexBuffer(drawBuffers[m_resourceIx].gpuDrawBuffers.indexBuffer.get(), 0u, asset::EIT_32BIT);
		cb->bindGraphicsPipeline(graphicsPipeline.get());
		cb->drawIndexed(currentIndexCount, 1u, 0u, 0u, 0u);

		if (fragmentShaderInterlockEnabled)
		{
			cb->bindDescriptorSets(asset::EPBP_GRAPHICS, resolveAlphaPipeLayout.get(), 0u, 1u, &descriptorSets[m_resourceIx].get());
			cb->bindGraphicsPipeline(resolveAlphaGraphicsPipeline.get());
			nbl::ext::FullScreenTriangle::recordDrawCalls(resolveAlphaGraphicsPipeline, 0u, swapchain->getPreTransform(), cb.get());
		}

		if constexpr (DebugMode)
		{
			cb->bindDescriptorSets(asset::EPBP_GRAPHICS, graphicsPipelineLayout.get(), 0u, 1u, &descriptorSets[m_resourceIx].get());
			cb->bindGraphicsPipeline(debugGraphicsPipeline.get());
			cb->drawIndexed(currentIndexCount, 1u, 0u, 0u, 0u);
		}
		cb->endQuery(pipelineStatsPool.get(), 0);
		cb->endRenderPass();

		cb->endDebugMarker();
		cb->end();

	}

	video::IGPUQueue::SSubmitInfo addObjects(video::IGPUQueue* submissionQueue, video::IGPUFence* submissionFence, video::IGPUQueue::SSubmitInfo& intendedNextSubmit)
	{
		// we record upload of our objects and if we failed to allocate we submit everything
		if (!intendedNextSubmit.isValid() || intendedNextSubmit.commandBufferCount <= 0u)
		{
			// log("intendedNextSubmit is invalid.", nbl::system::ILogger::ELL_ERROR);
			assert(false);
			return intendedNextSubmit;
		}

		// Use the last command buffer in intendedNextSubmit, it should be in recording state
		auto& cmdbuf = intendedNextSubmit.commandBuffers[intendedNextSubmit.commandBufferCount - 1];

		assert(cmdbuf->getState() == video::IGPUCommandBuffer::ES_RECORDING && cmdbuf->isResettable());
		assert(cmdbuf->getRecordingFlags().hasFlags(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT));

		auto* cmdpool = cmdbuf->getPool();
		assert(cmdpool->getQueueFamilyIndex() == submissionQueue->getFamilyIndex());

		auto& currentDrawBuffers = drawBuffers[m_resourceIx];
		currentDrawBuffers.setSubmitDrawsFunction(
			[&](video::IGPUQueue* submissionQueue, video::IGPUFence* submissionFence, video::IGPUQueue::SSubmitInfo intendedNextSubmit)
			{
				return submitInBetweenDraws(m_resourceIx, submissionQueue, submissionFence, intendedNextSubmit);
			}
		);
		currentDrawBuffers.reset();

		if constexpr (mode == ExampleMode::CASE_0)
		{
			CPULineStyle style = {};
			style.screenSpaceLineWidth = 0.0f;
			style.worldSpaceLineWidth = 5.0f;
			style.color = float32_t4(0.7f, 0.3f, 0.1f, 0.5f);

			CPolyline polyline;
			{
				std::vector<float64_t2> linePoints;
				linePoints.push_back({ -50.0, -50.0 });
				linePoints.push_back({ 50.0, 50.0 });
				polyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
			}

			intendedNextSubmit = currentDrawBuffers.drawPolyline(polyline, style, UseDefaultClipProjectionIdx, submissionQueue, submissionFence, intendedNextSubmit);
		}
		else if (mode == ExampleMode::CASE_1)
		{
			CPULineStyle style = {};
			style.screenSpaceLineWidth = 0.0f;
			style.worldSpaceLineWidth = 0.8f;
			style.color = float32_t4(0.619f, 0.325f, 0.709f, 0.2f);

			CPULineStyle style2 = {};
			style2.screenSpaceLineWidth = 0.0f;
			style2.worldSpaceLineWidth = 0.8f;
			style2.color = float32_t4(0.119f, 0.825f, 0.709f, 0.5f);

			intendedNextSubmit = currentDrawBuffers.drawPolyline(bigPolyline, style, UseDefaultClipProjectionIdx, submissionQueue, submissionFence, intendedNextSubmit);
			intendedNextSubmit = currentDrawBuffers.drawPolyline(bigPolyline2, style2, UseDefaultClipProjectionIdx, submissionQueue, submissionFence, intendedNextSubmit);
		}
		else if (mode == ExampleMode::CASE_2)
		{
			CPULineStyle style = {};
			style.screenSpaceLineWidth = 0.0f;
			style.worldSpaceLineWidth = 0.8f;
			style.color = float32_t4(0.619f, 0.325f, 0.709f, 0.9f);

			CPolyline polyline;
			std::vector<QuadraticBezierInfo> beziers;
			beziers.push_back({
				100.0 * float64_t2(-0.4, 0.13),
				100.0 * float64_t2(7.7, 3.57),
				100.0 * float64_t2(8.8, 7.27) });
			beziers.push_back({
				100.0 * float64_t2(6.6, 0.17),
				100.0 * float64_t2(-1.97, 3.2),
				100.0 * float64_t2(3.7, 7.27)});
			polyline.addQuadBeziers(core::SRange<QuadraticBezierInfo>(beziers.data(), beziers.data() + beziers.size()));

			core::SRange<CPolyline> polylines = core::SRange<CPolyline>(&polyline, &polyline + 1);
			auto debug = [&](CPolyline polyline, CPULineStyle lineStyle)
			{
				intendedNextSubmit = currentDrawBuffers.drawPolyline(polyline, lineStyle, UseDefaultClipProjectionIdx, submissionQueue, submissionFence, intendedNextSubmit);
			};

			Hatch hatch(polylines, HatchMajorAxis, nullptr);
			intendedNextSubmit = currentDrawBuffers.drawHatch(hatch, style, UseDefaultClipProjectionIdx, submissionQueue, submissionFence, intendedNextSubmit);
		}
		else if (mode == ExampleMode::CASE_3)
		{
			CPULineStyle style = {};
			style.screenSpaceLineWidth = 4.0f;
			style.worldSpaceLineWidth = 0.0f;
			style.color = float32_t4(0.7f, 0.3f, 0.1f, 0.5f);

			CPULineStyle style2 = {};
			style2.screenSpaceLineWidth = 5.0f;
			style2.worldSpaceLineWidth = 0.0f;
			style2.color = float32_t4(0.2f, 0.6f, 0.2f, 0.5f);


			CPolyline polyline;
			CPolyline polyline2;
			
			{

				float Left = -100;
				float Right = 100;
				float Base = -25;
				srand(95);
				std::vector<QuadraticBezierInfo> quadBeziers;
				for (int i = 0; i < 1; i++) {
					QuadraticBezierInfo quadratic1;
					quadratic1.p[0] = float64_t2(-100, 0);
					quadratic1.p[1] = float64_t2(-20, 0);
					quadratic1.p[2] = float64_t2(100, 0);
					quadBeziers.push_back(quadratic1);
				}
				
				//{
				//	QuadraticBezierInfo quadratic1;
				//	quadratic1.p[0] = float64_t2(50,0);
				//	quadratic1.p[1] = float64_t2(50,100);
				//	quadratic1.p[2] = float64_t2(100,100);
				//	quadBeziers.push_back(quadratic1);
				//}
				//{
				//	QuadraticBezierInfo quadratic1;
				//	quadratic1.p[0] = float64_t2(100, 100);
				//	quadratic1.p[1] = float64_t2(200, -200);
				//	quadratic1.p[2] = float64_t2(300, 300);
				//	quadBeziers.push_back(quadratic1);
				//}
				polyline.addQuadBeziers(core::SRange<QuadraticBezierInfo>(quadBeziers.data(), quadBeziers.data() + quadBeziers.size()));
			}
			{
				std::vector<QuadraticBezierInfo> quadBeziers;
				{
					QuadraticBezierInfo quadratic1;
					quadratic1.p[0] = float64_t2(0.0, 0.0);
					quadratic1.p[1] = float64_t2(20.0, 50.0);
					quadratic1.p[2] = float64_t2(80.0, 0.0);
					//quadBeziers.push_back(quadratic1);
				}
				{
					QuadraticBezierInfo quadratic1;
					quadratic1.p[0] = float64_t2(80.0, 0.0);
					quadratic1.p[1] = float64_t2(220.0, 50.0);
					quadratic1.p[2] = float64_t2(180.0, 200.0);
					//quadBeziers.push_back(quadratic1);
				}
				{
					QuadraticBezierInfo quadratic1;
					quadratic1.p[0] = float64_t2(180.0, 200.0);
					quadratic1.p[1] = float64_t2(-20.0, 100.0);
					quadratic1.p[2] = float64_t2(30.0, -50.0);
					//quadBeziers.push_back(quadratic1);
				}

				// TODO: Test this after sdf fixes, cause a linear bezier is causing problems (I think)
				// MixedParabola myCurve = MixedParabola::fromFourPoints(float64_t2(-60.0, 90.0), float64_t2(0.0, 0.0), float64_t2(50.0, 0.0), float64_t2(60.0,-20.0));
				// error = 1e-2
				// 
				// ExplicitEllipse myCurve = ExplicitEllipse(20.0, 50.0);
				// MixedCircle myCurve = MixedCircle::fromFourPoints(float64_t2(90.0, 20.0), float64_t2(-50, 0.0), float64_t2(50.0, 0.0), float64_t2(60.0, 40.0));
				// Parabola myCurve = Parabola::fromThreePoints(float64_t2(-6.0, 4.0), float64_t2(0.0, 0.0), float64_t2(5.0, 0.0));
				MixedParabola myCurve = MixedParabola::fromFourPoints(float64_t2(-60.0, 90.0), float64_t2(0.0, 0.0), float64_t2(50.0, 0.0), float64_t2(60.0,-20.0));

				AddBezierFunc addToBezier = [&](const QuadraticBezierInfo& info) -> void
					{
						quadBeziers.push_back(info);
					};

				static int ix = 0;
				ix++;

				const int pp = (ix / 30) % 8;

				double error = pow(10.0, -1.0 * double(pp + 1));

				adaptiveSubdivision(myCurve, 0.0, 50.0, 1e-2, addToBezier, 10u);

				polyline2.addQuadBeziers(core::SRange<QuadraticBezierInfo>(quadBeziers.data(), quadBeziers.data() + quadBeziers.size()));
			}

			//intendedNextSubmit = currentDrawBuffers.drawPolyline(polyline, style, UseDefaultClipProjectionIdx, submissionQueue, submissionFence, intendedNextSubmit);
			intendedNextSubmit = currentDrawBuffers.drawPolyline(polyline2, style2, UseDefaultClipProjectionIdx, submissionQueue, submissionFence, intendedNextSubmit);

			ClipProjectionData customClipProject = {};
			customClipProject.projectionToNDC = m_Camera.constructViewProjection(m_timeElapsed);
			customClipProject.projectionToNDC[0][0] *= 1.003f;
			customClipProject.projectionToNDC[1][1] *= 1.003f;
			customClipProject.maxClipNDC = float32_t2(0.5, 0.5);
			customClipProject.minClipNDC = float32_t2(-0.5, -0.5);
			uint32_t clipProjIdx = InvalidClipProjectionIdx;
			// intendedNextSubmit = currentDrawBuffers.addClipProjectionData_SubmitIfNeeded(customClipProject, clipProjIdx, submissionQueue, submissionFence, intendedNextSubmit);
			//intendedNextSubmit = currentDrawBuffers.drawPolyline(polyline, style2, clipProjIdx, submissionQueue, submissionFence, intendedNextSubmit);
		}
		else if (mode == ExampleMode::CASE_4)
		{
			constexpr uint32_t CURVE_CNT = 15u;
			constexpr uint32_t SPECIAL_CASE_CNT = 5u;

			CPULineStyle cpuLineStyle;
			cpuLineStyle.screenSpaceLineWidth = 5.0f;
			cpuLineStyle.worldSpaceLineWidth = 0.0f;
			cpuLineStyle.color = float32_t4(0.0f, 0.3f, 0.0f, 0.5f);

			std::vector<CPULineStyle> cpuLineStyles(CURVE_CNT, cpuLineStyle);
			std::vector<CPolyline> polylines(CURVE_CNT);

			{
				std::vector<QuadraticBezierInfo> quadratics(CURVE_CNT);

				// setting controll points
				{
					float64_t2 P0(-90, 68);
					float64_t2 P1(-41, 118);
					float64_t2 P2(88, 19);

					const float64_t2 translationVector(0, -5);

					uint32_t curveIdx = 0;
					while(curveIdx < CURVE_CNT - SPECIAL_CASE_CNT)
					{
						quadratics[curveIdx].p[0] = P0;
						quadratics[curveIdx].p[1] = P1;
						quadratics[curveIdx].p[2] = P2;

						P0 += translationVector;
						P1 += translationVector;
						P2 += translationVector;

						curveIdx++;
					}

					// special case 0 (line, evenly spaced points)
					const double prevLineLowestY = quadratics[curveIdx - 1].p[2].y;
					double lineY = prevLineLowestY - 10.0;

					quadratics[curveIdx].p[0] = float64_t2(-100, lineY);
					quadratics[curveIdx].p[1] = float64_t2(0, lineY);
					quadratics[curveIdx].p[2] = float64_t2(100, lineY);
					cpuLineStyles[curveIdx].color = float32_t4(0.7f, 0.3f, 0.1f, 0.5f);

					// special case 1 (line, not evenly spaced points)
					lineY -= 10.0;
					curveIdx++;

					quadratics[curveIdx].p[0] = float64_t2(-100, lineY);
					quadratics[curveIdx].p[1] = float64_t2(20, lineY);
					quadratics[curveIdx].p[2] = float64_t2(100, lineY);

					// special case 2 (folded line)
					lineY -= 10.0;
					curveIdx++;

					/*quadratics[curveIdx].p[0] = double2(-100, lineY);
					quadratics[curveIdx].p[1] = double2(200, lineY);
					quadratics[curveIdx].p[2] = double2(100, lineY);*/

					quadratics[curveIdx].p[0] = float64_t2(-10000, lineY);
					quadratics[curveIdx].p[1] = float64_t2(210, lineY);
					quadratics[curveIdx].p[2] = float64_t2(10000, lineY);

					// special case 3 (A.x == 0)
					curveIdx++;
					quadratics[curveIdx].p[0] = float64_t2(0.0, 0.0);
					quadratics[curveIdx].p[1] = float64_t2(3.0, 4.14);
					quadratics[curveIdx].p[2] = float64_t2(6.0, 4.0);
					cpuLineStyles[curveIdx].color = float32_t4(0.7f, 0.3f, 0.1f, 0.5f);

						// make sure A.x == 0
					float64_t2 A = quadratics[curveIdx].p[0] - 2.0 * quadratics[curveIdx].p[1] + quadratics[curveIdx].p[2];
					assert(A.x == 0);

					// special case 4 (symetric parabola)
					curveIdx++;
					quadratics[curveIdx].p[0] = float64_t2(-150.0, 20.0);
					quadratics[curveIdx].p[1] = float64_t2(150.0, 0.0);
					quadratics[curveIdx].p[2] = float64_t2(-150.0, -20.0);
					cpuLineStyles[curveIdx].color = float32_t4(0.7f, 0.3f, 0.1f, 0.5f);
				}

				std::array<core::vector<float>, CURVE_CNT> stipplePatterns;

				// TODO: fix uninvited circles at beggining and end of curves, solve with clipping (precalc tMin, tMax)

					// test case 0: test curve
				//stipplePatterns[0] = { 5.0f, -5.0f, 1.0f, -5.0f };
				stipplePatterns[0] = { 0.0f, -5.0f, 2.0f, -5.0f };
					// test case 1: lots of redundant values, should look exactly like stipplePattern[0]
				stipplePatterns[1] = { 1.0f, 2.0f, 2.0f, -4.0f, -1.0f, 1.0f, -3.0f, -1.5f, -0.3f, -0.2f }; 
					// test case 2:stipplePattern[0] but shifted curve but shifted to left by 2.5f
				stipplePatterns[2] = { 2.5f, -5.0f, 1.0f, -5.0f, 2.5f };
					// test case 3: starts and ends with negative value, stipplePattern[2] reversed (I'm suspisious about that, need more testing)
				stipplePatterns[3] = { -2.5f, 5.0f, -1.0f, 5.0f, -2.5f };
					// test case 4: starts with "don't draw section"
				stipplePatterns[4] = { -5.0f, 5.0f };
					// test case 5: invisible curve (shouldn't be send to GPU)
				stipplePatterns[5] = { -1.0f };
					// test case 6: invisible curve (shouldn't be send to GPU)
				stipplePatterns[6] = { -1.0f, -5.0f, -10.0f };
					// test case 7: continous curuve
				stipplePatterns[7] = { 25.0f, 25.0f };
					// test case 8: empty pattern (draw line with no pattern)
				stipplePatterns[8] = {};
					// test case 9: max pattern size
				stipplePatterns[9] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -2.0f };
					// test case 10: A = 0 (line), evenly distributed controll points (doesn't work)
				stipplePatterns[10] = { 5.0f, -5.0f, 1.0f, -5.0f };
					// test case 11: A = 0 (line), not evenly distributed controll points
				stipplePatterns[11] = { 5.0f, -5.0f, 1.0f, -5.0f };
					// test case 12: A = 0 (line), folds itself
				stipplePatterns[12] = { 5.0f, -5.0f, 1.0f, -5.0f };
					// test case 13: curve with A.x = 0
				//stipplePatterns[13] = { 0.5f, -0.5f, 0.1f, -0.5f };
				stipplePatterns[13] = { 0.0f, -0.5f, 0.2f, -0.5f };
					// test case 14: long parabola
				stipplePatterns[14] = { 5.0f, -5.0f, 1.0f, -5.0f };

				std::vector<uint32_t> activIdx = { 10 };
				for (uint32_t i = 0u; i < CURVE_CNT; i++)
				{
					cpuLineStyles[i].setStipplePatternData(nbl::core::SRange<float>(stipplePatterns[i].begin()._Ptr, stipplePatterns[i].end()._Ptr));
					polylines[i].addQuadBeziers(nbl::core::SRange<QuadraticBezierInfo>(& quadratics[i], &quadratics[i] + 1));

					activIdx.push_back(i);
					if (std::find(activIdx.begin(), activIdx.end(), i) == activIdx.end())
						cpuLineStyles[i].stipplePatternSize = -1;
				}
			}
			for (uint32_t i = 0u; i < CURVE_CNT; i++)
				intendedNextSubmit = currentDrawBuffers.drawPolyline(polylines[i], cpuLineStyles[i], UseDefaultClipProjectionIdx, submissionQueue, submissionFence, intendedNextSubmit);
		}

		intendedNextSubmit = currentDrawBuffers.finalizeAllCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
		return intendedNextSubmit;
	}

	video::IGPUQueue::SSubmitInfo submitInBetweenDraws(uint32_t resourceIdx, video::IGPUQueue* submissionQueue, video::IGPUFence* submissionFence, video::IGPUQueue::SSubmitInfo intendedNextSubmit)
	{
		// Use the last command buffer in intendedNextSubmit, it should be in recording state
		auto& cmdbuf = intendedNextSubmit.commandBuffers[intendedNextSubmit.commandBufferCount - 1];

		auto& currentDrawBuffers = drawBuffers[resourceIdx];

		uint32_t windowWidth = swapchain->getCreationParameters().width;
		uint32_t windowHeight = swapchain->getCreationParameters().height;

		nbl::video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
		{
			VkRect2D area;
			area.offset = { 0,0 };
			area.extent = { windowWidth, windowHeight };

			beginInfo.clearValueCount = 0u;
			beginInfo.framebuffer = framebuffersDynArraySmartPtr->begin()[m_SwapchainImageIx];
			beginInfo.renderpass = renderpassInBetween;
			beginInfo.renderArea = area;
			beginInfo.clearValues = nullptr;
		}

		asset::SViewport vp;
		vp.minDepth = 1.f;
		vp.maxDepth = 0.f;
		vp.x = 0u;
		vp.y = 0u;
		vp.width = windowWidth;
		vp.height = windowHeight;
		cmdbuf->setViewport(0u, 1u, &vp);

		VkRect2D scissor;
		scissor.extent = { windowWidth, windowHeight };
		scissor.offset = { 0, 0 };
		cmdbuf->setScissor(0u, 1u, &scissor);

		pipelineBarriersBeforeDraw(cmdbuf);

		cmdbuf->beginRenderPass(&beginInfo, asset::ESC_INLINE);

		const uint32_t currentIndexCount = drawBuffers[resourceIdx].getIndexCount();
		cmdbuf->bindDescriptorSets(asset::EPBP_GRAPHICS, graphicsPipelineLayout.get(), 0u, 1u, &descriptorSets[resourceIdx].get());
		cmdbuf->bindIndexBuffer(drawBuffers[resourceIdx].gpuDrawBuffers.indexBuffer.get(), 0u, asset::EIT_32BIT);
		cmdbuf->bindGraphicsPipeline(graphicsPipeline.get());
		cmdbuf->drawIndexed(currentIndexCount, 1u, 0u, 0u, 0u);

		if constexpr (DebugMode)
		{
			cmdbuf->bindGraphicsPipeline(debugGraphicsPipeline.get());
			cmdbuf->drawIndexed(currentIndexCount, 1u, 0u, 0u, 0u);
		}
		
		cmdbuf->endRenderPass();

		cmdbuf->end();

		video::IGPUQueue::SSubmitInfo submit = intendedNextSubmit;
		submit.signalSemaphoreCount = 0u;
		submit.pSignalSemaphores = nullptr;
		assert(submit.isValid());
		submissionQueue->submit(1u, &submit, submissionFence);
		intendedNextSubmit.commandBufferCount = 1u;
		intendedNextSubmit.commandBuffers = &cmdbuf;
		intendedNextSubmit.waitSemaphoreCount = 0u;
		intendedNextSubmit.pWaitSemaphores = nullptr;
		intendedNextSubmit.pWaitDstStageMask = nullptr;
		// we can reset the fence and commandbuffer because we fully wait for the GPU to finish here
		logicalDevice->blockForFences(1u, &submissionFence);
		logicalDevice->resetFences(1u, &submissionFence);
		cmdbuf->reset(video::IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
		cmdbuf->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);

		// reset things
		// currentDrawBuffers.clear();

		return intendedNextSubmit;
	}

	double dt = 0;
	double m_timeElapsed = 0.0;
	std::chrono::steady_clock::time_point lastTime;

	void workLoopBody() override
	{
		m_resourceIx++;
		if (m_resourceIx >= FRAMES_IN_FLIGHT)
			m_resourceIx = 0;

		auto now = std::chrono::high_resolution_clock::now();
		dt = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastTime).count();
		lastTime = now;
		m_timeElapsed += dt;

		if constexpr (mode == ExampleMode::CASE_0)
		{
			m_Camera.setSize(20.0 + abs(cos(m_timeElapsed * 0.001)) * 600);
		}

		inputSystem->getDefaultMouse(&mouse);
		inputSystem->getDefaultKeyboard(&keyboard);

		mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void
			{
				m_Camera.mouseProcess(events);
			}
		, logger.get());
		keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void
			{
				m_Camera.keyboardProcess(events);
			}
		, logger.get());

		auto& cb = m_cmdbuf[m_resourceIx];
		auto& fence = m_frameComplete[m_resourceIx];

		auto& graphicsQueue = queues[CommonAPI::InitOutput::EQT_GRAPHICS];

		nbl::video::IGPUQueue::SSubmitInfo submit;
		submit.commandBufferCount = 1u;
		submit.commandBuffers = &cb.get();
		submit.signalSemaphoreCount = 1u;
		submit.pSignalSemaphores = &m_renderFinished[m_resourceIx].get();
		nbl::video::IGPUSemaphore* waitSemaphores[1u] = { m_imageAcquire[m_resourceIx].get() };
		asset::E_PIPELINE_STAGE_FLAGS waitStages[1u] = { nbl::asset::EPSF_COLOR_ATTACHMENT_OUTPUT_BIT };
		submit.waitSemaphoreCount = 1u;
		submit.pWaitSemaphores = waitSemaphores;
		submit.pWaitDstStageMask = waitStages;

		beginFrameRender();

		submit = addObjects(graphicsQueue, fence.get(), submit);

		endFrameRender();

		graphicsQueue->submit(1u, &submit, fence.get());

		CommonAPI::Present(
			logicalDevice.get(),
			swapchain.get(),
			queues[CommonAPI::InitOutput::EQT_GRAPHICS],
			m_renderFinished[m_resourceIx].get(),
			m_SwapchainImageIx);

		getAndLogQueryPoolResults();
	}

	bool keepRunning() override
	{
		return windowCb->isWindowOpen();
	}
};

//NBL_COMMON_API_MAIN(CADApp)
int main(int argc, char** argv) {
	CommonAPI::main<CADApp>(argc, argv);
}