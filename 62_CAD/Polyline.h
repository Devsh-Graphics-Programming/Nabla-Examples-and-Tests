#pragma once

#include <nabla.h>
#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include <nbl/builtin/hlsl/math/geometry.hlsl>
#include <nbl/builtin/hlsl/shapes/util.hlsl>
#include "Curves.h"

struct PolylineSettings
{
	// Reset Line Style after gaps
	static constexpr bool ResetLineStyleOnDiscontinuity = false;
};

// holds values for `LineStyle` struct and caculates stipple pattern processed values, cant think of better name
// Also used for TextStyles aliased with some members here. (temporarily?)
struct LineStyleInfo
{
	static constexpr int32_t InvalidStipplePatternSize = -1;
	static constexpr double InvalidShapeOffset = nbl::hlsl::numeric_limits<double>::infinity;
	static constexpr double InvalidNormalizedShapeOffset = nbl::hlsl::numeric_limits<float>::infinity;
	static constexpr float PatternEpsilon = 1e-3f;  // TODO: I think for phase shift in normalized stipple space this is a reasonable value? right?
	static const uint32_t StipplePatternMaxSize = LineStyle::StipplePatternMaxSize;

	float32_t4 color = {};
	float screenSpaceLineWidth = 0.0f; // alternatively used as TextStyle::italicTiltSlope
	float worldSpaceLineWidth = 0.0f;  // alternatively used as TextStyle::boldInPixels
	
	/*
		Stippling Values:
	*/
	int32_t stipplePatternSize = 0u;
	float reciprocalStipplePatternLen = 0.0;
	float stipplePattern[StipplePatternMaxSize] = {};
	float phaseShift = 0.0;
	/*
		Customization Flags:
		isRoadStyleFlag -> generate mitering and use square edges in sdf
		stretchToFit -> allows stretching of the pattern to fit inside the current line or curve segment, this flag also cause each line or curve section to start from the beginning of the pattern
	*/
	bool isRoadStyleFlag = false;
	bool stretchToFit = false;
	/*
		special segment -> this is an index to stipple pattern to not stretch that value in the pattern when stretchToFit is true
		a valid shape segments means we don't want to stretch the shape segment and we treat it differently
		InvalidRigidSegmentIndex means we want to stretch the shape segment as well and we don't treat it differently -> simpler
	*/
	uint32_t rigidSegmentIdx = InvalidRigidSegmentIndex;
	

	// Cached and Precomputed Values used in preprocessing a polyline
	float shapeNormalizedPlaceInPattern = InvalidNormalizedShapeOffset;
	float rigidSegmentStart = 0.0f;
	float rigidSegmentEnd = 0.0f;
	float rigidSegmentLen = 0.0f;

	/*
	* stipplePatternUnnormalizedRepresentation is the pattern user fills, normalization makes the pattern size 1.0 with +,-,+,- pattern
	* shapeOffset is the offset into the unnormalized pattern a shape is going to draw, it's specialized this way because sometimes we don't want to stretch the pattern value the shape resides in.
	*/
	void setStipplePatternData(const std::span<const double> stipplePatternUnnormalizedRepresentation, double shapeOffsetInPattern = InvalidShapeOffset, bool stretch = false, bool rigidShapeSegment = false)
	{
		// Invalidate to possibly fill with correct values later
		shapeNormalizedPlaceInPattern = InvalidNormalizedShapeOffset;
		rigidSegmentIdx = InvalidRigidSegmentIndex;
		phaseShift = 0.0f;

		if (stipplePatternUnnormalizedRepresentation.size() == 0)
		{
			stipplePatternSize = 0;
			return;
		}

		nbl::core::vector<double> stipplePatternTransformed;

		// just to make sure we have a consistent definition of what's positive and what's negative
		auto isValuePositive = [](double x)
			{
				return (x >= 0);
			};

		// merge redundant values
		for (auto it = stipplePatternUnnormalizedRepresentation.begin(); it != stipplePatternUnnormalizedRepresentation.end();)
		{
			double redundantConsecutiveValuesSum = 0.0f;
			const bool firstValueIsPositive = isValuePositive(*it);
			do
			{
				redundantConsecutiveValuesSum += *it;
				it++;
			} while (it != stipplePatternUnnormalizedRepresentation.end() && (firstValueIsPositive == isValuePositive(*it)));

			stipplePatternTransformed.push_back(redundantConsecutiveValuesSum);
		}

		bool stippled = false; // has at least 1 draw and 1 gap section
		
		if (stipplePatternTransformed.size() != 1)
		{
			// merge first and last value if their sign matches
			double currentPhaseShift = 0.0f;
			const bool firstComponentPositive = isValuePositive(stipplePatternTransformed[0]);
			const bool lastComponentPositive = isValuePositive(stipplePatternTransformed[stipplePatternTransformed.size() - 1]);
			if (firstComponentPositive == lastComponentPositive)
			{
				currentPhaseShift += std::abs(stipplePatternTransformed[stipplePatternTransformed.size() - 1]);
				stipplePatternTransformed[0] += stipplePatternTransformed[stipplePatternTransformed.size() - 1];
				stipplePatternTransformed.pop_back();
			}
			
			assert(stipplePatternTransformed.size() <= StipplePatternMaxSize);

			if (stipplePatternTransformed.size() != 1)
			{
				stippled = true;

				// rotate values if first value is negative value
				if (!firstComponentPositive)
				{
					std::rotate(stipplePatternTransformed.rbegin(), stipplePatternTransformed.rbegin() + 1, stipplePatternTransformed.rend());
					currentPhaseShift += std::abs(stipplePatternTransformed[0]);
				}

				// calculate normalized prefix sum
				const uint32_t PREFIX_SUM_MAX_SZ = LineStyle::StipplePatternMaxSize - 1u;
				const uint32_t PREFIX_SUM_SZ = static_cast<uint32_t>(stipplePatternTransformed.size()) - 1;

				double prefixSum[PREFIX_SUM_MAX_SZ];
				prefixSum[0] = stipplePatternTransformed[0];

				for (uint32_t i = 1u; i < PREFIX_SUM_SZ; i++)
					prefixSum[i] = abs(stipplePatternTransformed[i]) + prefixSum[i - 1];

				const double rcpLen = 1.0 / (prefixSum[PREFIX_SUM_SZ - 1] + abs(stipplePatternTransformed[PREFIX_SUM_SZ]));

				for (uint32_t i = 0; i < PREFIX_SUM_SZ; i++)
					prefixSum[i] *= rcpLen;

				reciprocalStipplePatternLen = static_cast<float>(rcpLen);
				stipplePatternSize = PREFIX_SUM_SZ;
				for (uint32_t i = 0u; i < PREFIX_SUM_SZ; ++i)
					stipplePattern[i] = static_cast<float>(prefixSum[i]);

				currentPhaseShift = currentPhaseShift * rcpLen;

				if (stipplePatternUnnormalizedRepresentation[0] == 0.0)
					currentPhaseShift -= PatternEpsilon;
				else if (stipplePatternUnnormalizedRepresentation[0] < 0.0)
					currentPhaseShift += PatternEpsilon;

				phaseShift = static_cast<float>(currentPhaseShift);
			}
		}

		stretchToFit = stretch;

		// is all gap or all draw
		if (!stippled)
		{
			reciprocalStipplePatternLen = static_cast<float>(1.0 / abs(stipplePatternTransformed[0]));
			// all draw
			if (stipplePatternTransformed[0] >= 0.0)
			{
				stipplePattern[0] = 1.0;
				stipplePatternSize = 1;
			}
			// all gap
			else 
			{
				stipplePattern[0] = 0.0;
				stipplePatternSize = InvalidStipplePatternSize;
			}
		}
		
		if (shapeOffsetInPattern != InvalidShapeOffset)
		{
			setShapeOffset(shapeOffsetInPattern);
			if (rigidShapeSegment)
			{
				if (stippled)
				{
					rigidSegmentIdx = getPatternIdxFromNormalizedPosition(shapeNormalizedPlaceInPattern);
					// only way the stretchValue is going to change phase shift is if it's a non uniform stretch with a rigid segment (that one segment that shouldn't stretch)
					rigidSegmentStart = (rigidSegmentIdx >= 1u) ? stipplePattern[rigidSegmentIdx - 1u] : 0.0f;
					rigidSegmentEnd = (rigidSegmentIdx < stipplePatternSize) ? stipplePattern[rigidSegmentIdx] : 1.0f;
					rigidSegmentLen = rigidSegmentEnd - rigidSegmentStart;
				}
				else
				{
					rigidSegmentIdx = 0u;
					rigidSegmentLen = 1.0f;
				}
			}
		}
	}
		
	void scalePattern(float64_t scale)
	{
		reciprocalStipplePatternLen /= scale;
	}

	void setShapeOffset(float64_t offsetInWorldSpace)
	{
		shapeNormalizedPlaceInPattern = glm::fract(offsetInWorldSpace * reciprocalStipplePatternLen + phaseShift);
	}

	// If it's all draw or all gap
	inline bool isSingleSegment() const
	{
		const bool allDraw = stipplePattern[0] == 1.0f && stipplePatternSize == 1u;
		const bool allGap = !isVisible();
		return allDraw || allGap;
	}

	float calculateStretchValue(float64_t arcLen) const
	{
		float ret = 1.0f + LineStyleInfo::PatternEpsilon; // we stretch a little but more, this is to avoid clipped sdf numerical precision errors at the end of the line when we need it to be consistent (last pixels in a line or curve need to be in draw section or gap if end of pattern is in draw section or gap respectively)
		if (stretchToFit)
		{
			const bool singleRigidSegment = rigidShapeSegment() && isSingleSegment(); // we shouldn't apply any stretching if we only have one rigid stipple segment(either all draw or all gap(invisible)

			if (rigidShapeSegment() && arcLen * reciprocalStipplePatternLen <= rigidSegmentLen)
			{
				// arcLen is less than or equal to rigidSegmentLen, then stretch value is invalidated to draw the polyline solid without any style
				ret = InvalidStyleStretchValue;
			}
			else if (!singleRigidSegment)
			{
				//	Work out how many segments will fit into the line and calculate the stretch factor
				int nSegments = 1;
				double dInteger;
				double dFraction = ::modf(arcLen * reciprocalStipplePatternLen, &dInteger);
				if (dInteger < 1.0)
					nSegments = 1;
				else
				{
					nSegments = (int)dInteger;
					if (dFraction > 0.5)
						nSegments++;
				}

				if (dFraction == 0.0)
					ret = 1.0;
				else
				{
					// here we calculate the stretch value so that the pattern ends when the line/curve ends.
					// below `+ LineStyleInfo::PhaseShiftEpsilon`, stretches the value a little more  behaves as if arcLen is += epsilon * patternLen.
					// because we need it to be consistent (last pixels in a line or curve need to be in draw section or gap if end of pattern is in draw section or gap respectively)
					// example: when we need to fit a pattern onto a curve, and the pattern ends at a non-draw section
					//		the erros in the computations using phaseShift and stretch could incorrectly flag the last pixels of the curve to be in the pattern's next draw section but we needed it to end on a non draw section
					//		when stretching a little more it will ensure the clipping does it's job so that we don't get dots at the end of the line.
					ret = static_cast<float>((arcLen * reciprocalStipplePatternLen + LineStyleInfo::PatternEpsilon) / nSegments);
				}
			}
		}
		return ret;
	}

	// normalized place in pattern might change when stretching non-uniformly
	float getStretchedNormalizedPlaceInPattern(float32_t normalizedPlaceInPattern, float32_t stretchValue) const
	{
		if (stretchToFit && rigidShapeSegment() && !isSingleSegment())
		{
			float nonShapeSegmentStretchValue = (stretchValue - rigidSegmentLen) / (1.0 - rigidSegmentLen);
			float newPlaceInPattern = nbl::core::min(normalizedPlaceInPattern, rigidSegmentStart) * nonShapeSegmentStretchValue; // stretch parts before rigid segment
			newPlaceInPattern += nbl::core::max(normalizedPlaceInPattern - rigidSegmentEnd, 0.0f) * nonShapeSegmentStretchValue; // stretch parts after rigid segment
			newPlaceInPattern += nbl::core::max(nbl::core::min(rigidSegmentLen, normalizedPlaceInPattern - rigidSegmentStart), 0.0f); // stretch parts inside rigid segment
			newPlaceInPattern /= stretchValue; // scale back to normalized phaseShift
			return newPlaceInPattern;
		}
		else
		{
			return normalizedPlaceInPattern;
		}
	}

	float getStretchedPhaseShift(float32_t stretchValue) const
	{
		return getStretchedNormalizedPlaceInPattern(phaseShift, stretchValue);
	}
	
	float getStretchedShapeNormalizedPlaceInPattern(float32_t stretchValue) const
	{
		return getStretchedNormalizedPlaceInPattern(shapeNormalizedPlaceInPattern, stretchValue);
	}

	uint32_t getPatternIdxFromNormalizedPosition(float normalizedPlaceInPattern) const
	{
		float integralPart;
		normalizedPlaceInPattern = std::modf(normalizedPlaceInPattern, &integralPart);
		const float* patternPtr = std::upper_bound(stipplePattern, stipplePattern + stipplePatternSize, normalizedPlaceInPattern);
		return std::distance(stipplePattern, patternPtr);
	}

	LineStyle getAsGPUData() const
	{
		LineStyle ret = {};

		// pack into uint32_t
		for (uint32_t i = 0; i < stipplePatternSize; ++i)
		{
			const bool leftIsDot =
				(i > 1 && stipplePattern[i - 1] == stipplePattern[i - 2]) ||
				(i == 1 && stipplePattern[0] == 0.0);

			const bool rightIsDot =
				(i == stipplePatternSize && stipplePattern[0] == 0.0) ||
				(i + 2 <= stipplePatternSize && stipplePattern[i] == stipplePattern[i + 1]);

			ret.setStippleValue(i, stipplePattern[i]);

			if (leftIsDot)
				ret.stipplePattern[i] |= 1u << 30u;
			if (rightIsDot)
				ret.stipplePattern[i] |= 1u << 31u;
		}

		ret.color = color;
		ret.screenSpaceLineWidth = screenSpaceLineWidth;
		ret.worldSpaceLineWidth = worldSpaceLineWidth;
		ret.stipplePatternSize = stipplePatternSize;
		ret.reciprocalStipplePatternLen = reciprocalStipplePatternLen;
		ret.isRoadStyleFlag = isRoadStyleFlag;
		ret.rigidSegmentIdx = rigidSegmentIdx;

		return ret;
	}

	inline bool rigidShapeSegment() const { return rigidSegmentIdx != InvalidRigidSegmentIndex; }

	inline bool hasShape() const { return shapeNormalizedPlaceInPattern != InvalidNormalizedShapeOffset; }

	inline bool isVisible() const { return stipplePatternSize != InvalidStipplePatternSize; }

	inline bool skipPreprocess() const 
	{
		return isSingleSegment() && !hasShape() && !isRoadStyleFlag;
	}
};

class CPolylineBase
{
public:
	// each section consists of multiple connected lines or multiple connected ellipses
	struct SectionInfo
	{
		ObjectType	type;
		uint32_t	index; // can't make this a void* cause of vector resize
		uint32_t	count;
	};

	virtual size_t getSectionsCount() const = 0;
	virtual const SectionInfo& getSectionInfoAt(const uint32_t idx) const = 0;
	virtual const QuadraticBezierInfo& getQuadBezierInfoAt(const uint32_t idx) const = 0;
	virtual const LinePointInfo& getLinePointAt(const uint32_t idx) const = 0;
	virtual std::span<const PolylineConnector> getConnectors() const = 0;
	virtual bool checkSectionsContinuity(float64_t discontinuityErrorTolerance = 1e-5) const = 0;
	virtual bool checkSectionsActuallyClosed(float64_t discontinuityErrorTolerance = 1e-5) const = 0;
};

// It is not optimized because how you feed a Polyline to our cad renderer is your choice. this is just for convenience
// This is a Nabla Polyline used to feed to our CAD renderer. You can convert your Polyline to this class. or just use it directly.
class CPolyline : public CPolylineBase
{
public:
	CPolyline() :
		m_Min(float64_t2(nbl::hlsl::numeric_limits<float64_t>::max, nbl::hlsl::numeric_limits<float64_t>::max)),
		m_Max(float64_t2(nbl::hlsl::numeric_limits<float64_t>::lowest, nbl::hlsl::numeric_limits<float64_t>::lowest)),
		m_closedPolygon(false)
	{}

	size_t getSectionsCount() const override { return m_sections.size(); }

	const SectionInfo& getSectionInfoAt(const uint32_t idx) const override
	{
		return m_sections[idx];
	}

	const QuadraticBezierInfo& getQuadBezierInfoAt(const uint32_t idx) const override
	{
		return m_quadBeziers[idx];
	}

	const LinePointInfo& getLinePointAt(const uint32_t idx) const override
	{
		return m_linePoints[idx];
	}

	std::span<const PolylineConnector> getConnectors() const override
	{
		return m_polylineConnector;
	}
	
	// Check for Gap/Discontinuity, it won't affect complex styling but will affect the polyline offsetting, if you don't care about those then ignore checking for discontinuity.
	bool checkSectionsContinuity(float64_t discontinuityErrorTolerance = 1e-5) const override
	{
		const float64_t2 DiscontinuityErrorTolerance = float64_t2(discontinuityErrorTolerance, discontinuityErrorTolerance);
		// Check for continuity
		for (uint32_t i = 1; i < m_sections.size(); ++i)
		{
			const float64_t2 firstPoint = getSectionFirstPoint(m_sections[i]);
			const float64_t2 prevLastPoint = getSectionLastPoint(m_sections[i - 1u]);
			if (glm::any(glm::greaterThan(glm::abs(firstPoint - prevLastPoint), DiscontinuityErrorTolerance))) // checking individual components rather than dist.
				return false;
		}
		return true;
	}

	bool checkSectionsActuallyClosed(float64_t discontinuityErrorTolerance = 1e-5) const override
	{
		const float64_t2 DiscontinuityErrorTolerance = float64_t2(discontinuityErrorTolerance, discontinuityErrorTolerance);
		if (m_sections.size() > 0u)
		{
			const float64_t2 firstPoint = getSectionFirstPoint(m_sections.front());
			const float64_t2 lastPoint = getSectionLastPoint(m_sections.back());
			return glm::all(glm::lessThan(glm::abs(firstPoint - lastPoint), DiscontinuityErrorTolerance));
		}
		return true;
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

	void addLinePoints(const std::span<float64_t2> linePoints)
	{
		if (linePoints.size() <= 1u)
			return;

		const uint32_t oldLinePointSize = m_linePoints.size();
		const uint32_t newLinePointSize = oldLinePointSize + linePoints.size();
		m_linePoints.resize(newLinePointSize);
		for (uint32_t i = 0u; i < linePoints.size(); i++)
		{
			m_linePoints[oldLinePointSize + i].p = linePoints[i];
			m_linePoints[oldLinePointSize + i].phaseShift = 0.0;
			m_linePoints[oldLinePointSize + i].stretchValue = 1.0;
			addExtremum(linePoints[i]);
		}

		SectionInfo newSection = {};
		newSection.type = ObjectType::LINE;
		newSection.index = oldLinePointSize;
		newSection.count = static_cast<uint32_t>(linePoints.size() - 1u);
		m_sections.push_back(newSection);
	}

	void addEllipticalArcs(const std::span<curves::EllipticalArcInfo> ellipses, double errorThreshold)
	{
		nbl::core::vector<nbl::hlsl::shapes::QuadraticBezier<double>> beziersArray;
		for (const auto& ellipticalInfo : ellipses)
		{
			curves::Subdivision::AddBezierFunc addBeziers = [&](nbl::hlsl::shapes::QuadraticBezier<double>&& quadBezier)
				{
					beziersArray.push_back(quadBezier);
				};

			curves::Subdivision::adaptive(ellipticalInfo, errorThreshold, addBeziers);
			addQuadBeziers({ beziersArray.data(), beziersArray.data() + beziersArray.size() });
			beziersArray.clear();
		}
	}

	void addQuadBeziers(const std::span<nbl::hlsl::shapes::QuadraticBezier<double>> quadBeziers)
	{
		if (quadBeziers.empty())
			return;

		constexpr QuadraticBezierInfo EMPTY_QUADRATIC_BEZIER_INFO = {};
		const uint32_t oldQuadBezierSize = m_quadBeziers.size();
		const uint32_t newQuadBezierSize = oldQuadBezierSize + quadBeziers.size();
		m_quadBeziers.resize(newQuadBezierSize);
		for (uint32_t i = 0u; i < quadBeziers.size(); i++)
		{
			const uint32_t currBezierIdx = oldQuadBezierSize + i;
			m_quadBeziers[currBezierIdx].shape = quadBeziers[i];
			m_quadBeziers[currBezierIdx].phaseShift = 0.0;
			m_quadBeziers[currBezierIdx].stretchValue = 1.0;
			
			addExtremum(quadBeziers[i].P0);
			addExtremum(quadBeziers[i].P1); // Currently We add the control point instead of actual extremum to avoid computing a quadratic formula for each bezier we add
			addExtremum(quadBeziers[i].P2);
		}
				
		const bool unifiedSection = (m_sections.size() > lastSectionsSize && m_sections.back().type == ObjectType::QUAD_BEZIER);
		if (unifiedSection)
		{
			float64_t2 prevPoint = getSectionLastPoint(m_sections[m_sections.size() - 1u]);
			m_quadBeziers[oldQuadBezierSize].shape.P0 = prevPoint; // or we can average
			m_sections.back().count += static_cast<uint32_t>(quadBeziers.size());
		}
		else
		{
			SectionInfo newSection = {};
			newSection.type = ObjectType::QUAD_BEZIER;
			newSection.index = oldQuadBezierSize;
			newSection.count = static_cast<uint32_t>(quadBeziers.size());
			m_sections.push_back(newSection);
		}
	}

	// The next two functions make sure consecutive calls to `addQuadBeziers` between `beginUnifiedCurveSection` and `endUnifiedCurveSection` get into a single bezier section (to keep consistent with n4ce on polyline stretching on individual curve segments)
	void beginUnifiedCurveSection()
	{
		lastSectionsSize = m_sections.size();
	}
	void endUnifiedCurveSection()
	{
		lastSectionsSize = std::numeric_limits<uint32_t>::max();
	}

	typedef std::function<void(const float64_t2& /*position*/, const float64_t2& /*direction*/, float32_t /*stretch*/)> AddShapeFunc;

	void preprocessPolylineWithStyle(const LineStyleInfo& lineStyle, float64_t discontinuityErrorTolerance = 1e-5, const AddShapeFunc& addShape = {});

	float64_t2 getSectionFirstPoint(const SectionInfo& section) const
	{
		if (section.type == ObjectType::LINE)
		{
			const uint32_t firstLinePointIdx = section.index;
			return m_linePoints[firstLinePointIdx].p;
		}
		else if (section.type == ObjectType::QUAD_BEZIER)
		{
			const uint32_t firstBezierIdx = section.index;
			return m_quadBeziers[firstBezierIdx].shape.P0;
		}
		else
		{
			assert(false);
			return float64_t2{};
		}
	}

	float64_t2 getSectionLastPoint(const SectionInfo& section) const
	{
		if (section.type == ObjectType::LINE)
		{
			const uint32_t lastLinePointIdx = section.index + section.count;
			return m_linePoints[lastLinePointIdx].p;
		}
		else if (section.type == ObjectType::QUAD_BEZIER)
		{
			const uint32_t lastBezierIdx = section.index + section.count - 1u;
			return m_quadBeziers[lastBezierIdx].shape.P2;
		}
		else
		{
			assert(false);
			return float64_t2{};
		}
	}

	float64_t2 getSectionFirstTangent(const SectionInfo& section) const
	{
		if (section.type == ObjectType::LINE)
		{
			const uint32_t firstLinePointIdx = section.index;
			return m_linePoints[firstLinePointIdx + 1u].p - m_linePoints[firstLinePointIdx].p;
		}
		else if (section.type == ObjectType::QUAD_BEZIER)
		{
			const uint32_t firstBezierIdx = section.index;
			return m_quadBeziers[firstBezierIdx].shape.P1 - m_quadBeziers[firstBezierIdx].shape.P0;
		}
		else
		{
			assert(false);
			return float64_t2{};
		}
	}

	float64_t2 getSectionLastTangent(const SectionInfo& section) const
	{
		if (section.type == ObjectType::LINE)
		{
			const uint32_t lastLinePointIdx = section.index + section.count;
			return m_linePoints[lastLinePointIdx].p - m_linePoints[lastLinePointIdx - 1].p;
		}
		else if (section.type == ObjectType::QUAD_BEZIER)
		{
			const uint32_t lastBezierIdx = section.index + section.count - 1u;
			return m_quadBeziers[lastBezierIdx].shape.P2 - m_quadBeziers[lastBezierIdx].shape.P1;
		}
		else
		{
			assert(false);
			return float64_t2{};
		}
	}

	CPolyline generateParallelPolyline(float64_t offset, const float64_t maxError = 1e-5) const;
	
	// outputs two offsets to the polyline and connects the ends if not closed
	void makeWideWhole(CPolyline& outOffset1, CPolyline& outOffset2, float64_t offset, const float64_t maxError = 1e-5) const;

	// Manual CPU Styling: breaks the current polyline into more polylines based the stipple pattern
	// we could output a list/vector of polylines instead of using lambda but most of the time we need to work with the output and throw it away immediately.
	typedef std::function<void(const CPolyline& /*current stipple*/)> OutputPolylineFunc; 
	void stippleBreakDown(const LineStyleInfo& lineStyle, const OutputPolylineFunc& addPolyline, float64_t discontinuityErrorTolerance = 1e-5) const;

	void setClosed(bool closed)
	{
		m_closedPolygon = closed;
	}

	float64_t2 getMin() const { return m_Min; }
	float64_t2 getMax() const { return m_Max; }

	void transform(const float64_t2x2& rotScale, float64_t2 translate)
	{
		// transform is linear
		for (auto& linePoint : m_linePoints)
			linePoint.p = mul(rotScale,linePoint.p) + translate;
		for (auto& bezierPoint : m_quadBeziers)
		{
			bezierPoint.shape.P0 = mul(rotScale,bezierPoint.shape.P0) + translate;
			bezierPoint.shape.P1 = mul(rotScale,bezierPoint.shape.P1) + translate;
			bezierPoint.shape.P2 = mul(rotScale,bezierPoint.shape.P2) + translate;
		}

		// not useful for markers:
#if 0
		std::array<float64_t2, 4> corners = {
			float64_t2{ m_Min.x, m_Min.y },
			float64_t2{ m_Max.x, m_Min.y },
			float64_t2{ m_Min.x, m_Max.y },
			float64_t2{ m_Max.x, m_Max.y }
		};
	
		std::array<float64_t2, 4> transformedCorners;
		for (uint32_t i = 0; i < 4u; ++i)
			transformedCorners[i] = mul(rotScale, corners[i]) + translate;
	
		// Compute new AABB by finding min and max of transformed corners (OBB)
		m_Min = { nbl::hlsl::numeric_limits<float64_t>::max, nbl::hlsl::numeric_limits<float64_t>::max };
		m_Max = { nbl::hlsl::numeric_limits<float64_t>::lowest, nbl::hlsl::numeric_limits<float64_t>::lowest };

		for (const auto& corner : transformedCorners) {
			m_Min.x = nbl::core::min(m_Min.x, corner.x);
			m_Min.y = nbl::core::min(m_Min.y, corner.y);
			m_Max.x = nbl::core::max(m_Max.x, corner.x);
			m_Max.y = nbl::core::max(m_Max.y, corner.y);
		}
#endif
	}

protected:
	std::vector<PolylineConnector> m_polylineConnector;
	std::vector<SectionInfo> m_sections;
	std::vector<LinePointInfo> m_linePoints;
	std::vector<QuadraticBezierInfo> m_quadBeziers; // series of connected beziers, startig with P0 and ending with P1
	uint32_t lastSectionsSize = std::numeric_limits<uint32_t>::max();
	// important for miter and parallel generation
	bool m_closedPolygon = false;

	inline void addExtremum(const float64_t2& point) 
	{
		m_Min.x = nbl::hlsl::min(m_Min.x, point.x);
		m_Min.y = nbl::hlsl::min(m_Min.y, point.y);
		m_Max.x = nbl::hlsl::max(m_Max.x, point.x);
		m_Max.y = nbl::hlsl::max(m_Max.y, point.y);
	}

	// AABB
	float64_t2 m_Min; // min coordinate of the whole polyline
	float64_t2 m_Max; // max coordinate of the whole polyline

	// Next 3 are protected member functions to modify current lines and bezier sections used in polyline offsetting:

	void insertLinePointsToSection(uint32_t sectionIdx, uint32_t insertionPoint, const std::span<float64_t2> linePoints)
	{
		SectionInfo& section = m_sections[sectionIdx];
		assert(section.type == ObjectType::LINE);
		assert(insertionPoint <= section.count + 1u);

		const size_t addedCount = linePoints.size();

		constexpr LinePointInfo EMPTY_LINE_POINT_INFO = {};
		m_linePoints.insert(m_linePoints.begin() + section.index + insertionPoint, addedCount, EMPTY_LINE_POINT_INFO);
		for (uint32_t i = 0u; i < linePoints.size(); ++i)
			m_linePoints[section.index + insertionPoint + i].p = linePoints[i];

		if (section.count > 0)
			section.count += addedCount; // if it wasn't empty before, every new point constructs a new line
		else
			section.count += addedCount - 1u; // if it was empty before, every the first point doesn't create a new line only the next lines after that

		// update next sections offsets
		for (uint32_t i = sectionIdx + 1u; i < m_sections.size(); ++i)
		{
			if (m_sections[i].type == ObjectType::LINE)
				m_sections[i].index += addedCount;
		}
	}

	void removeSectionObjectsFromIdxToEnd(uint32_t sectionIdx, uint32_t idx)
	{
		if (sectionIdx >= m_sections.size())
			return;
		SectionInfo& section = m_sections[sectionIdx];
		if (idx >= section.count)
			return;

		const size_t removedCount = section.count - idx;
		if (section.type == ObjectType::LINE)
		{
			if (idx == 0) // if idx==0 it means it wants to delete the whole line section, so we remove all the line points including the first one in the section
				m_linePoints.erase(m_linePoints.begin() + section.index, m_linePoints.begin() + section.index + section.count + 1u);
			else
				m_linePoints.erase(m_linePoints.begin() + section.index + idx + 1u, m_linePoints.begin() + section.index + section.count + 1u);
		}
		else if (section.type == ObjectType::QUAD_BEZIER)
			m_quadBeziers.erase(m_quadBeziers.begin() + section.index + idx, m_quadBeziers.begin() + section.index + section.count);

		section.count -= removedCount;

		// update next sections offsets
		for (uint32_t i = sectionIdx + 1u; i < m_sections.size(); ++i)
		{
			if (m_sections[i].type == section.type)
				m_sections[i].index -= removedCount;
		}
	}

	void removeSectionObjectsFromBeginToIdx(uint32_t sectionIdx, uint32_t idx)
	{
		if (sectionIdx >= m_sections.size())
			return;
		SectionInfo& section = m_sections[sectionIdx];
		if (idx <= 0)
			return;
		const size_t removedCount = nbl::core::min(idx, section.count);
		if (section.type == ObjectType::LINE)
		{
			if (idx == section.count) // if idx==section.count it means it wants to delete the whole line section, so we remove all the line points including the last one in the section
				m_linePoints.erase(m_linePoints.begin() + section.index, m_linePoints.begin() + section.index + section.count + 1u);
			else
				m_linePoints.erase(m_linePoints.begin() + section.index, m_linePoints.begin() + section.index + idx);
		}
		else if (section.type == ObjectType::QUAD_BEZIER)
			m_quadBeziers.erase(m_quadBeziers.begin() + section.index, m_quadBeziers.begin() + section.index + idx);

		section.count -= removedCount;

		// update next sections offsets
		for (uint32_t i = sectionIdx + 1u; i < m_sections.size(); ++i)
		{
			if (m_sections[i].type == section.type)
				m_sections[i].index -= removedCount;
		}
	}

	struct SectionIntersectResult
	{
		static constexpr uint32_t InvalidIndex = ~0u;
		float64_t2 intersection;
		float64_t prevT; // for previous
		float64_t nextT; // for next
		uint32_t prevObjIndex = InvalidIndex;
		uint32_t nextObjIndex = InvalidIndex;

		bool valid() const { return prevObjIndex != InvalidIndex && nextObjIndex != InvalidIndex; }
		void invalidate()
		{
			prevObjIndex = InvalidIndex;
			nextObjIndex = InvalidIndex;
		}
	};

	SectionIntersectResult intersectLineSectionObjects(const SectionInfo& prevSection, uint32_t prevObjIdx, const SectionInfo& nextSection, uint32_t nextObjIdx) const
	{
		SectionIntersectResult res = {};
		res.invalidate();

		if (prevSection.type == ObjectType::LINE && nextSection.type == ObjectType::LINE)
		{
			const float64_t2 A0 = m_linePoints[prevSection.index + prevObjIdx].p;
			const float64_t2 V0 = m_linePoints[prevSection.index + prevObjIdx + 1u].p - A0;
			const float64_t2 A1 = m_linePoints[nextSection.index + nextObjIdx].p;
			const float64_t2 V1 = m_linePoints[nextSection.index + nextObjIdx + 1u].p - A1;

			const float64_t2 intersection = nbl::hlsl::shapes::util::LineLineIntersection<float64_t>(A0, V0, A1, V1);
			const float64_t t0 = nbl::hlsl::dot(V0, intersection - A0) / nbl::hlsl::dot(V0, V0);
			const float64_t t1 = nbl::hlsl::dot(V1, intersection - A1) / nbl::hlsl::dot(V1, V1);
			if (t0 >= 0.0 && t0 <= 1.0 && t1 >= 0.0 && t1 <= 1.0)
			{
				res.intersection = intersection;
				res.prevObjIndex = prevObjIdx;
				res.nextObjIndex = nextObjIdx;
				res.prevT = t0;
				res.nextT = t1;
			}
		}

		return res;
	}

	std::array<SectionIntersectResult, 2> intersectLineBezierSectionObjects(const SectionInfo& prevSection, uint32_t prevObjIdx, const SectionInfo& nextSection, uint32_t nextObjIdx) const
	{
		std::array<SectionIntersectResult, 2> res = {};
		res[0].invalidate();
		res[1].invalidate();

		if ((prevSection.type == ObjectType::QUAD_BEZIER && nextSection.type == ObjectType::LINE) || (prevSection.type == ObjectType::LINE && nextSection.type == ObjectType::QUAD_BEZIER))
		{
			const uint32_t lineIdx = (prevSection.type == ObjectType::LINE) ? prevSection.index + prevObjIdx : nextSection.index + nextObjIdx;
			const uint32_t bezierIdx = (prevSection.type == ObjectType::QUAD_BEZIER) ? prevSection.index + prevObjIdx : nextSection.index + nextObjIdx;
			const float64_t2 A = m_linePoints[lineIdx].p;
			const float64_t2 V = m_linePoints[lineIdx + 1u].p - A;

			const auto& bezier = m_quadBeziers[bezierIdx].shape;

			float64_t2 bezierTs = nbl::hlsl::shapes::getBezierLineIntersectionEquation<float64_t>(bezier, A, V).computeRoots();
			uint8_t resIdx = 0u;
			for (uint32_t i = 0u; i < 2u; ++i)
			{
				const float64_t bezierT = bezierTs[i];
				if (bezierT > 0.0 && bezierT < 1.0)
				{
					const float64_t2 intersection = bezier.evaluate(bezierT);
					const float64_t tLine = nbl::hlsl::dot(V, intersection - A) / nbl::hlsl::dot(V, V);

					if (tLine > 0.0 && tLine < 1.0)
					{
						auto& localRes = res[resIdx++];
						localRes.prevObjIndex = prevObjIdx;
						localRes.nextObjIndex = nextObjIdx;
						if (prevSection.type == ObjectType::LINE)
						{
							localRes.prevT = tLine;
							localRes.nextT = bezierT; // we don't care about lineT value
						}
						else
						{
							localRes.prevT = bezierT;
							localRes.nextT = tLine; // we don't care about lineT value
						}
						localRes.intersection = intersection;
					}
				}
			}
		}

		return res;
	}

	std::array<SectionIntersectResult, 4> intersectBezierBezierSectionObjects(const SectionInfo& prevSection, uint32_t prevObjIdx, const SectionInfo& nextSection, uint32_t nextObjIdx) const
	{
		std::array<SectionIntersectResult, 4> res = {};
		for (uint32_t i = 0u; i < 4u; ++i)
			res[i].invalidate();

		if (prevSection.type == ObjectType::QUAD_BEZIER && nextSection.type == ObjectType::QUAD_BEZIER)
		{
			const auto& prevBezier = m_quadBeziers[prevSection.index + prevObjIdx].shape;
			const auto& nextBezier = m_quadBeziers[nextSection.index + nextObjIdx].shape;
			float64_t4 bezierTs = nbl::hlsl::shapes::getBezierBezierIntersectionEquation<float64_t>(prevBezier, nextBezier).computeRoots();
			uint8_t resIdx = 0u;
			for (uint32_t i = 0u; i < 4u; ++i)
			{
				const float64_t prevBezierT = bezierTs[i];
				if (prevBezierT > 0.0 && prevBezierT < 1.0)
				{
					const float64_t2 intersection = prevBezier.evaluate(prevBezierT);
					
					// Optimization before doing SDF to find other T:
					// (for next bezier) If both P1 and the intersection point are on the same side of the P0 -> P2 line, it's a a valid intersection
					const bool sideP1 = nbl::hlsl::cross2D(nextBezier.P2 - nextBezier.P0, nextBezier.P1 - nextBezier.P0) >= 0.0;
					const bool sideIntersection = nbl::hlsl::cross2D(nextBezier.P2 - nextBezier.P0, intersection - nextBezier.P0) >= 0.0;
					if (sideP1 == sideIntersection)
					{
						auto& localRes = res[resIdx++];
						localRes.intersection = intersection;
						localRes.prevObjIndex = prevObjIdx;
						localRes.nextObjIndex = nextObjIdx;
						localRes.prevT = prevBezierT;
						localRes.nextT = nbl::hlsl::shapes::Quadratic<float64_t>::constructFromBezier(nextBezier).getClosestT(intersection); // basically doing sdf to find other T :D
					}
				}
			}
		}

		return res;
	}

	SectionIntersectResult intersectTwoSections(const SectionInfo& prevSection, const SectionInfo& nextSection) const
	{
		SectionIntersectResult ret = {};
		ret.invalidate();

		if (prevSection.count == 0 || nextSection.count == 0)
			return ret;

		const float64_t2 chordDir = glm::normalize(getSectionLastPoint(prevSection) - getSectionFirstPoint(prevSection));
		float64_t2x2 rotate = float64_t2x2({ chordDir.x, chordDir.y }, { -chordDir.y, chordDir.x });

		// Used for Sweep and Prune Algorithm
		struct SectionObject
		{
			float64_t start;
			float64_t end;
			uint32_t idxInSection;
			bool isInPrevSection;
		};

		std::vector<SectionObject> objs;
		auto addSectionToObjects = [&](const SectionInfo& section, bool isPrevSection)
			{
				for (uint32_t i = 0u; i < section.count; ++i)
				{
					SectionObject obj = {};
					obj.isInPrevSection = isPrevSection;
					obj.idxInSection = i;

					if (section.type == ObjectType::LINE)
					{
						float64_t2 P0 = mul(rotate, m_linePoints[section.index + i].p);
						float64_t2 P1 = mul(rotate, m_linePoints[section.index + i + 1u].p);
						obj.start = nbl::core::min(P0.x, P1.x);
						obj.end = nbl::core::max(P0.x, P1.x);
					}
					else if (section.type == ObjectType::QUAD_BEZIER)
					{
						float64_t2 P0 = mul(rotate, m_quadBeziers[section.index + i].shape.P0);
						float64_t2 P1 = mul(rotate, m_quadBeziers[section.index + i].shape.P1);
						float64_t2 P2 = mul(rotate, m_quadBeziers[section.index + i].shape.P2);
						const auto quadratic = nbl::hlsl::shapes::Quadratic<float64_t>::constructFromBezier(P0, P1, P2);
						const auto tExtremum = -quadratic.B.x / (2.0 * quadratic.A.x);

						obj.start = nbl::core::min(P0.x, P2.x);
						obj.end = nbl::core::max(P0.x, P2.x);

						if (tExtremum >= 0.0 && tExtremum <= 1.0)
						{
							float64_t xExtremum = quadratic.evaluate(tExtremum).x;
							obj.start = nbl::core::min(obj.start, xExtremum);
							obj.end = nbl::core::max(obj.end, xExtremum);
						}
					}
					objs.push_back(obj);
				}
			};

		addSectionToObjects(prevSection, true);
		addSectionToObjects(nextSection, false);
		if (objs.empty())
			return ret;

		std::stack<SectionObject> starts; // Next segments sorted by start points
		std::stack<float64_t> ends; // Next end points
		std::vector<SectionObject> activeCandidates;

		std::sort(objs.begin(), objs.end(), [&](const SectionObject& a, const SectionObject& b) { return a.start > b.start; });
		for (SectionObject& obj : objs)
			starts.push(obj);

		std::sort(objs.begin(), objs.end(), [&](const SectionObject& a, const SectionObject& b) { return a.end > b.end; });
		for (SectionObject& obj : objs)
			ends.push(obj.end);

		const float64_t maxValue = objs.front().end;

		int32_t currentIntersectionObjectRemoveCount = -1; // we use this value to select only one from many intersections that removes the most objects from both sections.
		auto addToCandidateSet = [&](const SectionObject& entry)
			{
				for (const auto& obj : activeCandidates)
				{
					if (obj.isInPrevSection != entry.isInPrevSection) // they should not be in the same section
					{
						uint32_t prevObjIdx = (obj.isInPrevSection) ? obj.idxInSection : entry.idxInSection;
						uint32_t nextObjIdx = (obj.isInPrevSection) ? entry.idxInSection : obj.idxInSection;

						int32_t objectsToRemove = (prevSection.count - prevObjIdx - 1u) + (nextObjIdx); // number of objects that will be pruned/removed if this intersection is selected
						assert(objectsToRemove >= 0);

						constexpr uint32_t MaxIntersectionResults = 4u;
						SectionIntersectResult intersectionResults[MaxIntersectionResults];
						uint32_t intersectionResultCount = 0u;

						if (prevSection.type == ObjectType::LINE && nextSection.type == ObjectType::LINE)
						{
							SectionIntersectResult localIntersectionResult = intersectLineSectionObjects(prevSection, prevObjIdx, nextSection, nextObjIdx);
							intersectionResults[0u] = localIntersectionResult;
							intersectionResultCount = 1u;
						}
						else if ((prevSection.type == ObjectType::QUAD_BEZIER && nextSection.type == ObjectType::LINE) || (prevSection.type == ObjectType::LINE && nextSection.type == ObjectType::QUAD_BEZIER))
						{
							std::array<SectionIntersectResult, 2> localIntersectionResults = intersectLineBezierSectionObjects(prevSection, prevObjIdx, nextSection, nextObjIdx);
							intersectionResults[0] = localIntersectionResults[0];
							intersectionResults[1] = localIntersectionResults[1];
							intersectionResultCount = 2u;
						}
						else if (prevSection.type == ObjectType::QUAD_BEZIER && nextSection.type == ObjectType::QUAD_BEZIER)
						{
							std::array<SectionIntersectResult, 4> localIntersectionResults = intersectBezierBezierSectionObjects(prevSection, prevObjIdx, nextSection, nextObjIdx);
							intersectionResults[0] = localIntersectionResults[0];
							intersectionResults[1] = localIntersectionResults[1];
							intersectionResults[2] = localIntersectionResults[2];
							intersectionResults[3] = localIntersectionResults[3];
							intersectionResultCount = 4u;
						}
						
						for (uint32_t i = 0u; i < intersectionResultCount; ++i)
						{
							if (intersectionResults[i].valid())
							{
								// TODO: Better Criterial to select between multiple intersections of the same objects
								if (objectsToRemove > currentIntersectionObjectRemoveCount)
								{
									ret = intersectionResults[i];
									currentIntersectionObjectRemoveCount = objectsToRemove;
								}
							}
						}
					}
				}
				activeCandidates.push_back(entry);
			};

		float64_t currentValue = starts.top().start;
		while (currentValue < maxValue)
		{
			double newValue = 0.0;

			const double nextEnd = ends.top();

			SectionObject nextObj = {};
			bool addNewObject = false;

			if (starts.empty())
				newValue = nextEnd;
			else
			{
				nextObj = starts.top();
				if (nextObj.start <= nextEnd)
				{
					newValue = nextObj.start;
					addNewObject = true;
				}
				else
					newValue = nextEnd;
			}
			if (addNewObject)
				starts.pop();
			else
				ends.pop();

			if (newValue > currentValue)
			{
				// advance and trim the candidate set
				auto oit = activeCandidates.begin();
				for (auto iit = activeCandidates.begin(); iit != activeCandidates.end(); iit++)
				{
					const double end = iit->end;
					if (newValue < end)
					{
						if (oit != iit)
							*oit = *iit;
						oit++;
					}
				}
				// trim
				const auto newSize = std::distance(activeCandidates.begin(), oit);
				activeCandidates.resize(newSize);

				currentValue = newValue;
			}

			if (addNewObject)
				addToCandidateSet(nextObj);
		}

		return ret;
	}

	bool checkIfInDrawSection(const LineStyleInfo& lineStyle, float normalizedPlaceInPattern)
	{
		const uint32_t patternIdx = lineStyle.getPatternIdxFromNormalizedPosition(normalizedPlaceInPattern);
		// odd patternIdx means a "no draw section" and current candidate should split into two nearest draw sections
		return !(patternIdx & 0x1);
	}

	void addMiterIfVisible(
		const float32_t2 prevLineNormal,
		const float32_t2 nextLineNormal,
		const float32_t2 center)
	{
		const float crossProductZ = nbl::hlsl::cross2D(nextLineNormal, prevLineNormal);
		constexpr float CROSS_PRODUCT_LINEARITY_EPSILON = 1.0e-6f;
		const bool isMiterVisible = std::abs(crossProductZ) >= CROSS_PRODUCT_LINEARITY_EPSILON;
		if (isMiterVisible)
		{
			const float64_t2 intersectionDirection = glm::normalize(prevLineNormal + nextLineNormal);
			const float64_t cosAngleBetweenNormals = glm::dot(prevLineNormal, nextLineNormal);

			PolylineConnector res = {
				.circleCenter = center,
				.v = static_cast<float32_t2>(intersectionDirection * std::sqrt(2.0 / (1.0 + cosAngleBetweenNormals))),
				.cosAngleDifferenceHalf = static_cast<float32_t>(std::sqrt((1.0 + cosAngleBetweenNormals) * 0.5)),
			};

			// need To flip direction?
			if (crossProductZ < 0.0f)
				res.v = -res.v;

			m_polylineConnector.push_back(res);
		}
	}
};