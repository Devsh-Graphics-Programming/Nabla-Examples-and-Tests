#pragma once

#include <nabla.h>
#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include <nbl/builtin/hlsl/math/geometry.hlsl>
#include <nbl/builtin/hlsl/shapes/util.hlsl>
#include "Curves.h"

// holds values for `LineStyle` struct and caculates stipple pattern re values, cant think of better name
// Also used for TextStyles aliased with some members here. (temporarily?)
struct LineStyleInfo
{
	static constexpr int32_t InvalidStipplePatternSize = -1;
	static constexpr double InvalidShapeOffset = nbl::hlsl::numeric_limits<double>::infinity;
	static constexpr double InvalidNormalizedShapeOffset = nbl::hlsl::numeric_limits<float>::infinity;
	static constexpr float PatternEpsilon = 1e-3f;  // TODO: I think for phase shift in normalized stipple space this is a reasonable value? right?
	static const uint32_t StipplePatternMaxSize = LineStyle::StipplePatternMaxSize;

	float32_t4 color;
	float screenSpaceLineWidth; // alternatively used as TextStyle::italicTiltSlope
	float worldSpaceLineWidth;  // alternatively used as TextStyle::boldInPixels
	
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

		assert(stipplePatternUnnormalizedRepresentation.size() <= StipplePatternMaxSize);

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
	virtual bool checkSectionsContinuity() const = 0;
};

// It is not optimized because how you feed a Polyline to our cad renderer is your choice. this is just for convenience
// This is a Nabla Polyline used to feed to our CAD renderer. You can convert your Polyline to this class. or just use it directly.
class CPolyline : public CPolylineBase
{
public:
	CPolyline() :
		m_Min(float64_t2(nbl::hlsl::numeric_limits<float64_t>::max, nbl::hlsl::numeric_limits<float64_t>::max)),
		m_Max(float64_t2(nbl::hlsl::numeric_limits<float64_t>::min, nbl::hlsl::numeric_limits<float64_t>::min)),
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

	bool checkSectionsContinuity() const override
	{
		// Check for continuity
		for (uint32_t i = 1; i < m_sections.size(); ++i)
		{
			constexpr float64_t POINT_EQUALITY_THRESHOLD = 1e-12;
			const float64_t2 firstPoint = getSectionFirstPoint(m_sections[i]);
			const float64_t2 prevLastPoint = getSectionLastPoint(m_sections[i - 1u]);
			if (glm::distance(firstPoint, prevLastPoint) > POINT_EQUALITY_THRESHOLD)
			{
				// DISCONNECTION DETECTED, will break styling and offsetting the polyline, if you don't care about those then ignore discontinuity.
				return false;
			}
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

	void preprocessPolylineWithStyle(const LineStyleInfo& lineStyle, const AddShapeFunc& addShape = {})
	{
		if (lineStyle.skipPreprocess())
			return;
		// DISCONNECTION DETECTED, will break styling and offsetting the polyline, if you don't care about those then ignore discontinuity.
		// _NBL_DEBUG_BREAK_IF(!checkSectionsContinuity());
		PolylineConnectorBuilder connectorBuilder;

		const bool shouldAddShapes = (lineStyle.hasShape() && addShape.operator bool());
		// When stretchToFit is true, the curve section and individual lines should start from the beginning of the pattern (phaseShift = lineStyle.phaseShift)
		const bool patternStartAgain = lineStyle.stretchToFit;
		float currentPhaseShift = lineStyle.phaseShift;

		for (uint32_t sectionIdx = 0u; sectionIdx < m_sections.size(); sectionIdx++)
		{
			const auto& section = m_sections[sectionIdx];

			if (section.type == ObjectType::LINE)
			{
				// calculate phase shift at each point of each line in section
				const uint32_t lineCount = section.count;
				for (uint32_t i = 0u; i < lineCount; i++)
				{
					const uint32_t currIdx = section.index + i;
					auto& linePoint = m_linePoints[currIdx];
					const auto& nextLinePoint = m_linePoints[currIdx + 1u];
					const float64_t2 lineVector = nextLinePoint.p - linePoint.p;
					const float64_t lineLen = glm::length(lineVector);
					const float32_t stretchValue = lineStyle.calculateStretchValue(lineLen);
					const float rcpStretchedPatternLen = (lineStyle.reciprocalStipplePatternLen) / stretchValue;

					if (patternStartAgain)
						currentPhaseShift = lineStyle.getStretchedPhaseShift(stretchValue);

					linePoint.phaseShift = currentPhaseShift;
					linePoint.stretchValue = stretchValue;
					
					if (lineStyle.isRoadStyleFlag)
						connectorBuilder.addLineNormal(lineVector, lineLen, linePoint.p, currentPhaseShift);

					if (shouldAddShapes)
					{
						float shapeOffsetNormalized = lineStyle.getStretchedShapeNormalizedPlaceInPattern(stretchValue);
						// next shape Offset from start of line/curve
						float nextShapeOffset = shapeOffsetNormalized - currentPhaseShift;
						if (nextShapeOffset < 0.0f)
							nextShapeOffset += 1.0f;
						
						int32_t numberOfShapes = static_cast<int32_t>(std::ceil(lineLen * rcpStretchedPatternLen - nextShapeOffset)); // numberOfShapes = (ArcLen - nextShapeOffset*PatternLen)/PatternLen + 1

						float64_t stretchedPatternLen = 1.0 / (float64_t)rcpStretchedPatternLen;
						float64_t currentWorldSpaceOffset = nextShapeOffset * stretchedPatternLen;
						float64_t2 direction = lineVector / lineLen;
						for (int32_t s = 0; s < numberOfShapes; ++s)
						{
							const float64_t2 position = linePoint.p + direction * currentWorldSpaceOffset;
							addShape(position, direction, stretchValue);
							currentWorldSpaceOffset += stretchedPatternLen;
						}
					}

					if (!patternStartAgain)
					{
						// setting next phase shift based on current arc length
						const double changeInPhaseShift = glm::fract(lineLen * rcpStretchedPatternLen);
						currentPhaseShift = static_cast<float32_t>(glm::fract(currentPhaseShift + changeInPhaseShift));
					}
				}
			}
			else if (section.type == ObjectType::QUAD_BEZIER)
			{
				const uint32_t quadBezierCount = section.count;
				
				// when stretchToFit is true, we need to calculate the whole section arc length to figure out the stretch value needed for stippling phaseshift
				float stretchValue = 1.0;
				if (lineStyle.stretchToFit)
				{
					float64_t sectionArcLen = 0.0;
					for (uint32_t i = 0u; i < quadBezierCount; i++)
					{
						const QuadraticBezierInfo& quadBezierInfo = m_quadBeziers[section.index + i];
						nbl::hlsl::shapes::Quadratic<double> quadratic = nbl::hlsl::shapes::Quadratic<double>::constructFromBezier(quadBezierInfo.shape);
						nbl::hlsl::shapes::Quadratic<double>::ArcLengthCalculator arcLenCalc = nbl::hlsl::shapes::Quadratic<double>::ArcLengthCalculator::construct(quadratic);
						sectionArcLen += arcLenCalc.calcArcLen(1.0);
					}
					stretchValue = lineStyle.calculateStretchValue(sectionArcLen);
				}
				
				if (patternStartAgain)
					currentPhaseShift = lineStyle.getStretchedPhaseShift(stretchValue);

				const float rcpStretchedPatternLen = (lineStyle.reciprocalStipplePatternLen) / stretchValue;

				// calculate phase shift at point P0 of each bezier
				for (uint32_t i = 0u; i < quadBezierCount; i++)
				{
					const uint32_t currIdx = section.index + i;

					QuadraticBezierInfo& quadBezierInfo = m_quadBeziers[currIdx];
					quadBezierInfo.phaseShift = currentPhaseShift;
					quadBezierInfo.stretchValue = stretchValue;

					if (lineStyle.isRoadStyleFlag)
						connectorBuilder.addBezierNormals(m_quadBeziers[currIdx], currentPhaseShift);
					
					// setting next phase shift based on current arc length
					nbl::hlsl::shapes::Quadratic<double> quadratic = nbl::hlsl::shapes::Quadratic<double>::constructFromBezier(quadBezierInfo.shape);
					nbl::hlsl::shapes::Quadratic<double>::ArcLengthCalculator arcLenCalc = nbl::hlsl::shapes::Quadratic<double>::ArcLengthCalculator::construct(quadratic);
					const double bezierLen = arcLenCalc.calcArcLen(1.0);
					
					if (shouldAddShapes)
					{
						float shapeOffsetNormalized = lineStyle.getStretchedShapeNormalizedPlaceInPattern(stretchValue);

						// next shape Offset from start of line/curve
						float nextShapeOffset = shapeOffsetNormalized - currentPhaseShift;
						if (nextShapeOffset < 0.0f)
							nextShapeOffset += 1.0f;
						
						int32_t numberOfShapes = static_cast<int32_t>(std::ceil(bezierLen * rcpStretchedPatternLen - nextShapeOffset)); // numberOfShapes = (ArcLen - nextShapeOffset*PatternLen)/PatternLen + 1

						float64_t stretchedPatternLen = 1.0 / (float64_t)rcpStretchedPatternLen;
						float64_t currentWorldSpaceOffset = nextShapeOffset * stretchedPatternLen;
						for (int32_t s = 0; s < numberOfShapes; ++s)
						{
							float64_t t = arcLenCalc.calcArcLenInverse(quadratic, 0.0, 1.0, currentWorldSpaceOffset, 1e-5, 0.5);
							addShape(quadratic.evaluate(t), quadratic.derivative(t), stretchValue);
							currentWorldSpaceOffset += stretchedPatternLen;
						}
					}

					const double changeInPhaseShift = glm::fract(bezierLen * rcpStretchedPatternLen);
					currentPhaseShift = static_cast<float32_t>(glm::fract(currentPhaseShift + changeInPhaseShift));
				}
			}
		}

		if (lineStyle.isRoadStyleFlag)
		{
			connectorBuilder.setPhaseShiftAtEndOfPolyline(currentPhaseShift);
			m_polylineConnector = connectorBuilder.buildConnectors(lineStyle, m_closedPolygon);
		}
	}

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

	CPolyline generateParallelPolyline(float64_t offset, const float64_t maxError = 1e-5) const
	{
		// DISCONNECTION DETECTED, will break styling and offsetting the polyline, if you don't care about those then ignore discontinuity.
		_NBL_DEBUG_BREAK_IF(!checkSectionsContinuity());
		
		// TODO: move to nbl::hlsl later on as convenience function, correctly templated
		auto safe_normalize = [](nbl::hlsl::float64_t2 vec) -> nbl::hlsl::float64_t2
			{
				const float64_t dirLen = glm::length(vec);
				if (dirLen == 0.0)
					return nbl::hlsl::float64_t2(0.0, 0.0);
				return vec / dirLen;
			};

		CPolyline parallelPolyline = {};
		parallelPolyline.setClosed(m_closedPolygon);

		// The next two lamda functions connect offsetted beziers and lines
		// This function adds a bezier section and connects it to the previous section
		auto& newSections = parallelPolyline.m_sections;

		constexpr uint32_t InvalidSectionIdx = ~0u;
		uint32_t previousLineSectionIdx = InvalidSectionIdx;
		constexpr float64_t CROSS_PRODUCT_LINEARITY_EPSILON = 1e-5;
		auto connectTwoSections = [&](uint32_t prevSectionIdx, uint32_t nextSectionIdx)
			{
				auto& prevSection = newSections[prevSectionIdx];
				auto& nextSection = newSections[nextSectionIdx];

				float64_t2 prevTangent = parallelPolyline.getSectionLastTangent(prevSection);
				float64_t2 nextTangent = parallelPolyline.getSectionFirstTangent(nextSection);
				const float64_t crossProduct = nbl::hlsl::cross2D(prevTangent, nextTangent);

				if (abs(crossProduct) > CROSS_PRODUCT_LINEARITY_EPSILON)
				{
					if (crossProduct * offset > 0u) // Outward, needs connection
					{
						float64_t2 prevSectionEndPos = parallelPolyline.getSectionLastPoint(prevSection);
						float64_t2 nextSectionStartPos = parallelPolyline.getSectionFirstPoint(nextSection);
						float64_t2 intersection = nbl::hlsl::shapes::util::LineLineIntersection<float64_t>(prevSectionEndPos, prevTangent, nextSectionStartPos, nextTangent);

						if (nextSection.type == ObjectType::LINE)
						{
							if (prevSection.type == ObjectType::LINE)
							{
								// Change last point of left segment and first point of right segment to be equal to their intersection.
								parallelPolyline.m_linePoints[prevSection.index + prevSection.count].p = intersection;
								parallelPolyline.m_linePoints[nextSection.index].p = intersection;
							}
							else if (prevSection.type == ObjectType::QUAD_BEZIER)
							{
								// Add QuadBez Position + Intersection to start of right segment
								float64_t2 newLinePoints[2u] = { prevSectionEndPos, intersection };
								parallelPolyline.insertLinePointsToSection(nextSectionIdx, 0u, newLinePoints);
							}
						}
						else if (nextSection.type == ObjectType::QUAD_BEZIER)
						{
							if (prevSection.type == ObjectType::LINE)
							{
								// Add Intersection + right Segment first line position to end of leftSegment
								float64_t2 newLinePoints[2u] = { intersection, nextSectionStartPos };
								parallelPolyline.insertLinePointsToSection(prevSectionIdx, prevSection.count + 1u, newLinePoints);
							}
							else if (prevSection.type == ObjectType::QUAD_BEZIER)
							{
								// Add Intersection + right Segment first line position to end of leftSegment
								float64_t2 newLinePoints[3u] = { prevSectionEndPos, intersection, nextSectionStartPos };

								uint32_t linePointsInsertion = 0u;
								if (previousLineSectionIdx != InvalidSectionIdx)
									linePointsInsertion = newSections[previousLineSectionIdx].index + newSections[previousLineSectionIdx].count + 1u;

								const uint32_t newSectionIdx = prevSectionIdx + 1u;
								SectionInfo newSection = {};
								newSection.type = ObjectType::LINE;
								newSection.index = linePointsInsertion;
								newSection.count = 0u;
								newSections.insert(newSections.begin() + newSectionIdx, newSection);
								parallelPolyline.insertLinePointsToSection(newSectionIdx, 0u, newLinePoints);
								previousLineSectionIdx = newSectionIdx;
							}
						}
					}
					else // Inward Needs Trim and Prune
					{
						SectionIntersectResult sectionIntersectResult = parallelPolyline.intersectTwoSections(prevSection, nextSection);

						if (sectionIntersectResult.valid())
						{
							const bool sameSectionIntersection = (prevSectionIdx == nextSectionIdx);
							if (sameSectionIntersection)
								std::swap(sectionIntersectResult.prevObjIndex, sectionIntersectResult.nextObjIndex); // because `intersectionTwoSections` function prioritizes the largest distance intersection, it will get the indices swapped to get the largest indices distance
									
							assert(sectionIntersectResult.prevObjIndex < prevSection.count);
							assert(sectionIntersectResult.nextObjIndex < nextSection.count);
							
							// we want to first delete from (idx to end) and then (begin to idx)
							parallelPolyline.removeSectionObjectsFromIdxToEnd(prevSectionIdx, sectionIntersectResult.prevObjIndex + 1u);
							parallelPolyline.removeSectionObjectsFromBeginToIdx(nextSectionIdx, sectionIntersectResult.nextObjIndex);
							const bool removePrevSection = (prevSection.count == 0u);
							const bool removeNextSection = (nextSection.count == 0u);

							if (nextSection.type == ObjectType::LINE)
							{
								if (prevSection.type == ObjectType::LINE)
								{
									if (!removePrevSection)
										parallelPolyline.m_linePoints[prevSection.index + prevSection.count].p = sectionIntersectResult.intersection;
									if (!removeNextSection)
										parallelPolyline.m_linePoints[nextSection.index].p = sectionIntersectResult.intersection;
								}
								else if (prevSection.type == ObjectType::QUAD_BEZIER)
								{
									if (!removePrevSection)
										parallelPolyline.m_quadBeziers[prevSection.index + prevSection.count - 1u].shape.splitFromStart(sectionIntersectResult.prevT);
									if (!removeNextSection)
										parallelPolyline.m_linePoints[nextSection.index].p = sectionIntersectResult.intersection;
								}
							}
							else if (nextSection.type == ObjectType::QUAD_BEZIER)
							{
								if (prevSection.type == ObjectType::LINE)
								{
									if (!removePrevSection)
										parallelPolyline.m_linePoints[prevSection.index + prevSection.count].p = sectionIntersectResult.intersection;
									if (!removeNextSection)
										parallelPolyline.m_quadBeziers[nextSection.index].shape.splitToEnd(sectionIntersectResult.nextT);
								}
								else if (prevSection.type == ObjectType::QUAD_BEZIER)
								{
									if (!removePrevSection)
										parallelPolyline.m_quadBeziers[prevSection.index + prevSection.count - 1u].shape.splitFromStart(sectionIntersectResult.prevT);
									if (!removeNextSection)
										parallelPolyline.m_quadBeziers[nextSection.index].shape.splitToEnd(sectionIntersectResult.nextT);
								}
							}
							
							// Remove Sections that got their whole objects removed
							if (nextSectionIdx >= prevSectionIdx)
							{
								// we want to first delete the higher idx 
								if (removeNextSection)
									parallelPolyline.m_sections.erase(parallelPolyline.m_sections.begin() + nextSectionIdx);
								if (removePrevSection && !sameSectionIntersection)
									parallelPolyline.m_sections.erase(parallelPolyline.m_sections.begin() + prevSectionIdx);
							}
							else
							{
								if (removePrevSection)
									parallelPolyline.m_sections.erase(parallelPolyline.m_sections.begin() + prevSectionIdx);
								if (removeNextSection)
									parallelPolyline.m_sections.erase(parallelPolyline.m_sections.begin() + nextSectionIdx);
							}
						}
						else
						{
							// TODO: If Polyline is continuous, and the tangents are reported correctly this shouldn't happen
						}
					}
				}
			};
		auto connectBezierSection = [&](std::vector<nbl::hlsl::shapes::QuadraticBezier<double>>&& beziers)
			{
				parallelPolyline.addQuadBeziers(beziers);
				// If there is a previous section, connect to that
				if (newSections.size() > 1u)
				{
					const uint32_t prevSectionIdx = newSections.size() - 2u;
					const uint32_t nextSectionIdx = newSections.size() - 1u;
					connectTwoSections(prevSectionIdx, nextSectionIdx);
				}
			};
		auto connectLinesSection = [&](std::vector<float64_t2>&& linePoints)
			{
				parallelPolyline.addLinePoints(linePoints);
				// If there is a previous section, connect to that
				if (newSections.size() > 1u)
				{
					const uint32_t prevSectionIdx = newSections.size() - 2u;
					const uint32_t nextSectionIdx = newSections.size() - 1u;
					connectTwoSections(prevSectionIdx, nextSectionIdx);
				}
				previousLineSectionIdx = newSections.size() - 1u;
			};

		// This loop Generates Mitered Line Sections and Offseted Beziers -> will still have breaks and disconnections 
		// then we call addBezierSection and it connects it to the previous section
		for (uint32_t i = 0; i < m_sections.size(); ++i)
		{
			const auto& section = m_sections[i];
			if (section.type == ObjectType::LINE)
			{
				// TODO: try merging lines if they have same tangent (resultin in less points)
				std::vector<float64_t2> newLinePoints;
				newLinePoints.reserve(m_linePoints.size());
				for (uint32_t j = 0; j < section.count + 1; ++j)
				{
					const uint32_t linePointIdx = section.index + j;
					float64_t2 offsetVector;
					if (j == 0)
					{
						const float64_t2 tangent = safe_normalize(m_linePoints[linePointIdx + 1].p - m_linePoints[linePointIdx].p);
						offsetVector = float64_t2(tangent.y, -tangent.x);
					}
					else if (j == section.count)
					{
						const float64_t2 tangent = safe_normalize(m_linePoints[linePointIdx].p - m_linePoints[linePointIdx - 1].p);
						offsetVector = float64_t2(tangent.y, -tangent.x);
					}
					else
					{
						const float64_t2 tangentPrevLine = safe_normalize(m_linePoints[linePointIdx].p - m_linePoints[linePointIdx - 1].p);
						const float64_t2 normalPrevLine = float64_t2(tangentPrevLine.y, -tangentPrevLine.x);
						const float64_t2 tangentNextLine = safe_normalize(m_linePoints[linePointIdx + 1].p - m_linePoints[linePointIdx].p);
						const float64_t2 normalNextLine = float64_t2(tangentNextLine.y, -tangentNextLine.x);

						const float64_t2 intersectionDirection = safe_normalize(normalPrevLine + normalNextLine);
						const float64_t cosAngleBetweenNormals = glm::dot(normalPrevLine, normalNextLine);
						offsetVector = intersectionDirection * sqrt(2.0 / (1.0 + cosAngleBetweenNormals));
					}
					newLinePoints.push_back(m_linePoints[linePointIdx].p + offsetVector * offset);
				}
				connectLinesSection(std::move(newLinePoints));
			}
			else if (section.type == ObjectType::QUAD_BEZIER)
			{
				std::vector<nbl::hlsl::shapes::QuadraticBezier<double>> newBeziers;
				curves::Subdivision::AddBezierFunc addToBezier = [&](nbl::hlsl::shapes::QuadraticBezier<double>&& info) -> void
					{
						newBeziers.push_back(info);
					};
				for (uint32_t j = 0; j < section.count; ++j)
				{
					const uint32_t bezierIdx = section.index + j;
					curves::OffsettedBezier offsettedBezier(m_quadBeziers[bezierIdx].shape, offset);
					curves::Subdivision::adaptive(offsettedBezier, maxError, addToBezier, 10u);
				}
				connectBezierSection(std::move(newBeziers));
			}
		}

		if (parallelPolyline.m_closedPolygon)
		{
			const uint32_t prevSectionIdx = newSections.size() - 1u;
			const uint32_t nextSectionIdx = 0u;
			connectTwoSections(prevSectionIdx, nextSectionIdx);
		}

		return parallelPolyline;
	}
	
	// outputs two offsets to the polyline and connects the ends if not closed
	void makeWideWhole(CPolyline& outOffset1, CPolyline& outOffset2, float64_t offset, const float64_t maxError = 1e-5) const
	{
		outOffset1 = generateParallelPolyline(offset, maxError);
		outOffset2 = generateParallelPolyline(-1.0 * offset, maxError);

		if (!m_closedPolygon)
		{
			nbl::hlsl::float64_t2 beginToBeginConnector[2u];
			beginToBeginConnector[0u] = outOffset1.getSectionFirstPoint(outOffset1.getSectionInfoAt(0u));
			beginToBeginConnector[1u] = outOffset2.getSectionFirstPoint(outOffset2.getSectionInfoAt(0u));
			nbl::hlsl::float64_t2 endToEndConnector[2u];
			endToEndConnector[0u] = outOffset1.getSectionLastPoint(outOffset1.getSectionInfoAt(outOffset1.getSectionsCount() - 1u));
			endToEndConnector[1u] = outOffset2.getSectionLastPoint(outOffset2.getSectionInfoAt(outOffset2.getSectionsCount() - 1u));
			outOffset2.addLinePoints({ beginToBeginConnector, beginToBeginConnector + 2 });
			outOffset2.addLinePoints({ endToEndConnector, endToEndConnector + 2 });
		}
	}
			
	// Manual CPU Styling: breaks the current polyline into more polylines based the stipple pattern
	// we could output a list/vector of polylines instead of using lambda but most of the time we need to work with the output and throw it away immediately.
	typedef std::function<void(const CPolyline& /*current stipple*/)> OutputPolylineFunc; 
	void stippleBreakDown(const LineStyleInfo& lineStyle, const OutputPolylineFunc& addPolyline) const
	{
		if (!lineStyle.isVisible())
			return;
		
		// currently only works for road styles with only 2 stipple values (1 draw, 1 gap)
		assert(lineStyle.stipplePatternSize == 1);
		
		const float64_t patternLen = 1.0 / lineStyle.reciprocalStipplePatternLen;
		const float32_t drawSectionNormalizedLen = lineStyle.stipplePattern[0];
		const float32_t gapSectionNormalizedLen = 1.0 - lineStyle.stipplePattern[0];
		const float32_t drawSectionLen = drawSectionNormalizedLen / lineStyle.reciprocalStipplePatternLen;
		
		CPolyline currentPolyline;
		std::vector<float64_t2> linePoints;
		std::vector<nbl::hlsl::shapes::QuadraticBezier<float64_t>> beziers;
		auto flushCurrentPolyline = [&]()
			{
				if (linePoints.size() > 1u)
				{
					currentPolyline.addLinePoints({ linePoints.data(), linePoints.data() + linePoints.size() });
					linePoints.clear();
				}
				if (beziers.size() > 0u)
				{
					currentPolyline.addQuadBeziers({ beziers.data(), beziers.data() + beziers.size() });
					beziers.clear();
				}
				if (currentPolyline.getSectionsCount() > 0u)
					addPolyline(currentPolyline);
				currentPolyline.clearEverything();
			};
		auto pushBackToLinePoints = [&](const float64_t2& point)
			{
				if (linePoints.empty())
				{
					linePoints.push_back(point);
				}
				else if (linePoints.back() != point)
				{
					linePoints.push_back(point);
				}
			};

		float currentPhaseShift = lineStyle.phaseShift;
		for (uint32_t sectionIdx = 0u; sectionIdx < m_sections.size(); sectionIdx++)
		{
			const auto& section = m_sections[sectionIdx];

			if (section.type == ObjectType::LINE)
			{
				// calculate phase shift at each point of each line in section
				const uint32_t lineCount = section.count;
				for (uint32_t i = 0u; i < lineCount; i++)
				{
					const uint32_t currIdx = section.index + i;
					const auto& currlinePoint = m_linePoints[currIdx];
					const auto& nextLinePoint = m_linePoints[currIdx + 1u];
					const float64_t2 lineVector = nextLinePoint.p - currlinePoint.p;
					const float64_t lineLen = glm::length(lineVector);
					const float64_t2 lineVectorNormalized = lineVector / lineLen;

					float64_t currentTracedLen = 0.0;
					const float32_t differenceToNextDrawSectionEnd = drawSectionNormalizedLen - currentPhaseShift;
					const bool insideDrawSection = differenceToNextDrawSectionEnd > 0.0f;
					if (insideDrawSection)
					{
						const float64_t nextDrawSectionEnd = differenceToNextDrawSectionEnd / lineStyle.reciprocalStipplePatternLen;
						const bool finishesOnThisShape = nextDrawSectionEnd <= lineLen;
						
						pushBackToLinePoints(currlinePoint.p);
						
						if (finishesOnThisShape)
						{
							pushBackToLinePoints(currlinePoint.p + nextDrawSectionEnd * lineVectorNormalized);
							flushCurrentPolyline();
						}
						else
						{
							pushBackToLinePoints(nextLinePoint.p);
						}
						currentTracedLen = nbl::core::min(nextDrawSectionEnd, lineLen);
					}
					
					const float32_t currentTracedLenPlaceInPattern = glm::fract(currentPhaseShift + currentTracedLen * lineStyle.reciprocalStipplePatternLen);
					const float32_t differenceToNextDrawSectionBegin = glm::fract(1.0 - currentTracedLenPlaceInPattern); // 0.0 gives 0.0, and 1.0 gives 0.0
					const float64_t lenToNextDrawBegin = differenceToNextDrawSectionBegin / lineStyle.reciprocalStipplePatternLen;
					currentTracedLen = nbl::core::min(currentTracedLen + lenToNextDrawBegin, lineLen);
					const float64_t remainingLen = lineLen - currentTracedLen;
					
					if (remainingLen > 0.0)
					{
						float64_t remainingLenNormalized = remainingLen * lineStyle.reciprocalStipplePatternLen;
						int32_t fullPatternsFitNumber = static_cast<int32_t>(std::ceil(remainingLenNormalized));
						linePoints.reserve(linePoints.size() + fullPatternsFitNumber * 2u);
						
						for (int32_t s = 0; s < fullPatternsFitNumber; ++s)
						{
							const bool completelyInsideShape = (currentTracedLen + drawSectionLen) <= lineLen;
							const float64_t nextDrawSectionEnd = (completelyInsideShape) ? (currentTracedLen + drawSectionLen) : lineLen;
							
							pushBackToLinePoints(currlinePoint.p + currentTracedLen * lineVectorNormalized);
							pushBackToLinePoints(currlinePoint.p + nextDrawSectionEnd * lineVectorNormalized);

							if (completelyInsideShape)
								flushCurrentPolyline();

							currentTracedLen = nbl::core::min(currentTracedLen + patternLen, lineLen);
						}
					}

					const double changeInPhaseShift = glm::fract(lineLen * lineStyle.reciprocalStipplePatternLen);
					currentPhaseShift = static_cast<float32_t>(glm::fract(currentPhaseShift + changeInPhaseShift));
				}
				
				currentPolyline.addLinePoints({ linePoints.data(), linePoints.data() + linePoints.size() });
				linePoints.clear();
			}
			else if (section.type == ObjectType::QUAD_BEZIER)
			{
				const uint32_t quadBezierCount = section.count;
				
				// calculate phase shift at point P0 of each bezier
				for (uint32_t i = 0u; i < quadBezierCount; i++)
				{
					const uint32_t currIdx = section.index + i;
					const QuadraticBezierInfo& quadBezierInfo = m_quadBeziers[currIdx];
					// setting next phase shift based on current arc length
					nbl::hlsl::shapes::Quadratic<double> quadratic = nbl::hlsl::shapes::Quadratic<double>::constructFromBezier(quadBezierInfo.shape);
					nbl::hlsl::shapes::Quadratic<double>::ArcLengthCalculator arcLenCalc = nbl::hlsl::shapes::Quadratic<double>::ArcLengthCalculator::construct(quadratic);
					const double bezierLen = arcLenCalc.calcArcLen(1.0);

					float64_t currentTracedLen = 0.0;
					const float32_t differenceToNextDrawSectionEnd = drawSectionNormalizedLen - currentPhaseShift;
					const bool insideDrawSection = differenceToNextDrawSectionEnd > 0.0f;
					if (insideDrawSection)
					{
						const float64_t nextDrawSectionEnd = differenceToNextDrawSectionEnd / lineStyle.reciprocalStipplePatternLen;
						const bool finishesOnThisShape = nextDrawSectionEnd <= bezierLen;
						
						auto newBezier = quadBezierInfo.shape;
						if (finishesOnThisShape)
						{
							float64_t t = arcLenCalc.calcArcLenInverse(quadratic, 0.0, 1.0, nextDrawSectionEnd, 1e-5, 0.5);
							newBezier.splitFromStart(t);
							beziers.push_back(std::move(newBezier));
							flushCurrentPolyline();
						}
						else
						{
							// draw section covers entire bezier, no need for clipping
							beziers.push_back(std::move(newBezier));
						}

						currentTracedLen = nbl::core::min(nextDrawSectionEnd, bezierLen);
					}
					
					const float32_t currentTracedLenPlaceInPattern = glm::fract(currentPhaseShift + currentTracedLen * lineStyle.reciprocalStipplePatternLen);
					const float32_t differenceToNextDrawSectionBegin = glm::fract(1.0 - currentTracedLenPlaceInPattern); // 0.0 gives 0.0, and 1.0 gives 0.0
					const float64_t lenToNextDrawBegin = differenceToNextDrawSectionBegin / lineStyle.reciprocalStipplePatternLen;
					currentTracedLen = nbl::core::min(currentTracedLen + lenToNextDrawBegin, bezierLen);
					const float64_t remainingLen = bezierLen - currentTracedLen;

					if (remainingLen > 0.0)
					{
						float64_t remainingLenNormalized = remainingLen * lineStyle.reciprocalStipplePatternLen;
						int32_t fullPatternsFitNumber = static_cast<int32_t>(std::ceil(remainingLenNormalized));
						beziers.reserve(beziers.size() + fullPatternsFitNumber);

						for (int32_t s = 0; s < fullPatternsFitNumber; ++s)
						{
							const bool completelyInsideShape = (currentTracedLen + drawSectionLen) <= bezierLen;
							const float64_t nextDrawSectionEnd = (completelyInsideShape) ? (currentTracedLen + drawSectionLen) : bezierLen;
							float64_t tStart = arcLenCalc.calcArcLenInverse(quadratic, 0.0, 1.0, currentTracedLen, 1e-5, 0.5);
							float64_t tEnd = arcLenCalc.calcArcLenInverse(quadratic, 0.0, 1.0, nextDrawSectionEnd, 1e-5, 0.5);

							auto newBezier = quadBezierInfo.shape;
							newBezier.splitFromMinToMax(tStart, tEnd);
							beziers.push_back(std::move(newBezier));
							
							if (completelyInsideShape)
								flushCurrentPolyline();

							currentTracedLen = nbl::core::min(currentTracedLen + patternLen, bezierLen);
						}
					}
					
					const double changeInPhaseShift = glm::fract(bezierLen * lineStyle.reciprocalStipplePatternLen);
					currentPhaseShift = static_cast<float32_t>(glm::fract(currentPhaseShift + changeInPhaseShift));
				}
				
				currentPolyline.addQuadBeziers({ beziers.data(), beziers.data() + beziers.size() });
				beziers.clear();
			}
		}

		flushCurrentPolyline();
	}

	void setClosed(bool closed)
	{
		m_closedPolygon = closed;
	}

	float64_t2 getMin() const { return m_Min; }
	float64_t2 getMax() const { return m_Max; }

protected:
	std::vector<PolylineConnector> m_polylineConnector;
	std::vector<SectionInfo> m_sections;
	std::vector<LinePointInfo> m_linePoints;
	std::vector<QuadraticBezierInfo> m_quadBeziers;
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

private:
	class PolylineConnectorBuilder
	{
	public:
		void addLineNormal(const float64_t2& line, float64_t lineLen, const float64_t2& worldSpaceCircleCenter, float phaseShift)
		{
			PolylineConnectorNormalHelperInfo connectorNormalInfo;
			connectorNormalInfo.worldSpaceCircleCenter = worldSpaceCircleCenter;
			connectorNormalInfo.normal = float64_t2(-line.y, line.x) / lineLen;
			connectorNormalInfo.phaseShift = phaseShift;
			connectorNormalInfo.type = ObjectType::LINE;

			connectorNormalInfos.push_back(connectorNormalInfo);
		}

		void addBezierNormals(const QuadraticBezierInfo& quadBezierInfo, float phaseShift)
		{
			// TODO: we already calculate quadratic form of each bezier (except of the last one), maybe store this info in an array and use it later to calculate normals?
			const float64_t2 bezierDerivativeValueAtP0 = 2.0 * (quadBezierInfo.shape.P1 - quadBezierInfo.shape.P0);
			const float32_t2 tangentAtP0 = glm::normalize(bezierDerivativeValueAtP0);
			//const float_t2 A = P0 - 2.0 * P1 + P2;
			const float64_t2 bezierDerivativeValueAtP2 = 2.0 * (quadBezierInfo.shape.P0 - 2.0 * quadBezierInfo.shape.P1 + quadBezierInfo.shape.P2) + 2.0 * (quadBezierInfo.shape.P1 - quadBezierInfo.shape.P0);
			const float32_t2 tangentAtP2 = glm::normalize(bezierDerivativeValueAtP2);

			PolylineConnectorNormalHelperInfo connectorNormalInfoAtP0{};
			connectorNormalInfoAtP0.worldSpaceCircleCenter = quadBezierInfo.shape.P0;
			connectorNormalInfoAtP0.normal = float32_t2(-tangentAtP0.y, tangentAtP0.x);
			connectorNormalInfoAtP0.phaseShift = phaseShift;
			connectorNormalInfoAtP0.type = ObjectType::QUAD_BEZIER;

			PolylineConnectorNormalHelperInfo connectorNormalInfoAtP2{};
			connectorNormalInfoAtP2.worldSpaceCircleCenter = quadBezierInfo.shape.P2;
			connectorNormalInfoAtP2.normal = float32_t2(-tangentAtP2.y, tangentAtP2.x);
			connectorNormalInfoAtP2.phaseShift = phaseShift;
			connectorNormalInfoAtP2.type = ObjectType::QUAD_BEZIER;

			connectorNormalInfos.push_back(connectorNormalInfoAtP0);
			connectorNormalInfos.push_back(connectorNormalInfoAtP2);
		}

		std::vector<PolylineConnector> buildConnectors(const LineStyleInfo& lineStyle, bool isClosedPolygon)
		{
			std::vector<PolylineConnector> connectors;

			if (connectorNormalInfos.size() < 2u)
				return {};

			uint32_t i = getConnectorNormalInfoCountOfLineType(connectorNormalInfos[0].type);
			while (i < connectorNormalInfos.size())
			{
				const auto& prevLine = connectorNormalInfos[i - 1];
				const auto& nextLine = connectorNormalInfos[i];
				constructMiterIfVisible(lineStyle, prevLine, nextLine, false, connectors);

				i += getConnectorNormalInfoCountOfLineType(nextLine.type);
			}

			if (isClosedPolygon)
			{
				const auto& prevLine = connectorNormalInfos[connectorNormalInfos.size() - 1u];
				const auto& nextLine = connectorNormalInfos[0u];
				constructMiterIfVisible(lineStyle, prevLine, nextLine, true, connectors);
			}

			return connectors;
		}

		inline void setPhaseShiftAtEndOfPolyline(float phaseShift)
		{
			phaseShiftAtEndOfPolyline = phaseShift;
		}

	private:
		struct PolylineConnectorNormalHelperInfo
		{
			float64_t2 worldSpaceCircleCenter;
			float32_t2 normal;
			float phaseShift;
			ObjectType type;
		};

		inline uint32_t getConnectorNormalInfoCountOfLineType(ObjectType type)
		{
			static constexpr uint32_t NORMAL_INFO_COUNT_OF_A_LINE = 1u;
			static constexpr uint32_t NORMAL_INFO_COUNT_OF_A_QUADRATIC_BEZIER = 2u;

			if (type == ObjectType::LINE)
			{
				return NORMAL_INFO_COUNT_OF_A_LINE;
			}
			else if (type == ObjectType::QUAD_BEZIER)
			{
				return NORMAL_INFO_COUNT_OF_A_QUADRATIC_BEZIER;
			}
			else
			{
				assert(false);
				return -1u;
			}
		}

		bool checkIfInDrawSection(const LineStyleInfo& lineStyle, float normalizedPlaceInPattern)
		{
			const uint32_t patternIdx = lineStyle.getPatternIdxFromNormalizedPosition(normalizedPlaceInPattern);
			// odd patternIdx means a "no draw section" and current candidate should split into two nearest draw sections
			return !(patternIdx & 0x1);
		}

		void constructMiterIfVisible(
			const LineStyleInfo& lineStyle,
			const PolylineConnectorNormalHelperInfo& prevLine,
			const PolylineConnectorNormalHelperInfo& nextLine,
			bool isMiterClosingPolyline,
			std::vector<PolylineConnector>& connectors)
		{
			const float32_t2 prevLineNormal = prevLine.normal;
			const float32_t2 nextLineNormal = nextLine.normal;

			const float crossProductZ = nbl::hlsl::cross2D(nextLineNormal, prevLineNormal);
			constexpr float CROSS_PRODUCT_LINEARITY_EPSILON = 1.0e-6f;
			const bool isMiterVisible = std::abs(crossProductZ) >= CROSS_PRODUCT_LINEARITY_EPSILON;
			bool isMiterInDrawSection = checkIfInDrawSection(lineStyle, nextLine.phaseShift);
			if (isMiterClosingPolyline)
			{
				isMiterInDrawSection = isMiterInDrawSection && checkIfInDrawSection(lineStyle, phaseShiftAtEndOfPolyline);
			}

			if (isMiterVisible && isMiterInDrawSection)
			{
				const float64_t2 intersectionDirection = glm::normalize(prevLineNormal + nextLineNormal);
				const float64_t cosAngleBetweenNormals = glm::dot(prevLineNormal, nextLineNormal);

				PolylineConnector res{};
				res.circleCenter = nextLine.worldSpaceCircleCenter;
				res.v = static_cast<float32_t2>(intersectionDirection * std::sqrt(2.0 / (1.0 + cosAngleBetweenNormals)));
				res.cosAngleDifferenceHalf = static_cast<float32_t>(std::sqrt((1.0 + cosAngleBetweenNormals) * 0.5));

				const bool needToFlipDirection = crossProductZ < 0.0f;
				if (needToFlipDirection)
				{
					res.v = -res.v;
				}
				// Negating y to avoid doing it in vertex shader when working in screen space, where y is in the opposite direction of worldspace y direction
				res.v.y = -res.v.y;

				connectors.push_back(res);
			}
		}

		nbl::core::vector<PolylineConnectorNormalHelperInfo> connectorNormalInfos;
		float phaseShiftAtEndOfPolyline = 0.0f;
	};
};
