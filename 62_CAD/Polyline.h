#pragma once

#include <nabla.h>
#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include <nbl/builtin/hlsl/math/geometry.hlsl>
#include <nbl/builtin/hlsl/shapes/util.hlsl>
#include "curves.h"

// holds values for `LineStyle` struct and caculates stipple pattern re values, cant think of better name
struct CPULineStyle
{
	static constexpr int32_t InvalidStipplePatternSize = -1;
	static const uint32_t STIPPLE_PATTERN_MAX_SZ = 15u;

	float32_t4 color;
	float screenSpaceLineWidth;
	float worldSpaceLineWidth;
	// gpu stipple pattern data form
	int32_t stipplePatternSize = 0u;
	float reciprocalStipplePatternLen;
	float stipplePattern[STIPPLE_PATTERN_MAX_SZ];
	float phaseShift;
	bool isRoadStyleFlag;

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

		// just to make sure we have a consistent definition of what's positive and what's negative
		auto isValuePositive = [](float x)
		{
			return (x >= 0);
		};

		// merge redundant values
		for (auto it = stipplePatternCPURepresentation.begin(); it != stipplePatternCPURepresentation.end();)
		{
			float redundantConsecutiveValuesSum = 0.0f;
			const bool firstValueSign = isValuePositive(*it);
			do
			{
				redundantConsecutiveValuesSum += *it;
				it++;
			} while (it != stipplePatternCPURepresentation.end() && (firstValueSign == isValuePositive(*it)));

			stipplePatternTransformed.push_back(redundantConsecutiveValuesSum);
		}

		if (stipplePatternTransformed.size() == 1)
		{
			stipplePatternSize = stipplePatternTransformed[0] < 0.0f ? InvalidStipplePatternSize : 0;
			return;
		}

		// merge first and last value if their sign matches
		phaseShift = 0.0f;
		const bool firstComponentPositive = isValuePositive(stipplePatternTransformed[0]);
		const bool lastComponentPositive = isValuePositive(stipplePatternTransformed[stipplePatternTransformed.size() - 1]);
		if (firstComponentPositive == lastComponentPositive)
		{
			phaseShift += std::abs(stipplePatternTransformed[stipplePatternTransformed.size() - 1]);
			stipplePatternTransformed[0] += stipplePatternTransformed[stipplePatternTransformed.size() - 1];
			stipplePatternTransformed.pop_back();
		}

		if (stipplePatternTransformed.size() == 1)
		{
			stipplePatternSize = (firstComponentPositive) ? 0 : InvalidStipplePatternSize;
			return;
		}

		// rotate values if first value is negative value
		if (!firstComponentPositive)
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

		reciprocalStipplePatternLen = 1.0f / (prefixSum[PREFIX_SUM_SZ - 1] + abs(stipplePatternTransformed[PREFIX_SUM_SZ]));

		for (int i = 0; i < PREFIX_SUM_SZ; i++)
			prefixSum[i] *= reciprocalStipplePatternLen;

		stipplePatternSize = PREFIX_SUM_SZ;
		std::memcpy(stipplePattern, prefixSum, sizeof(prefixSum));

		phaseShift = phaseShift * reciprocalStipplePatternLen;
		if (stipplePatternTransformed[0] == 0.0)
		{
			phaseShift -= 1.0e-3f; // TODO: I think 1e-3 phase shift in normalized stipple space is a reasonable value? right?
		}
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

			ret.stipplePattern[i] = static_cast<uint32_t>(stipplePattern[i] * (1u << 29u));

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

		return ret;
	}

	inline bool isVisible() const { return stipplePatternSize != InvalidStipplePatternSize; }
};

static_assert(sizeof(DrawObject) == 16u);
static_assert(sizeof(MainObject) == 8u);
static_assert(sizeof(Globals) == 128u);
static_assert(sizeof(LineStyle) == 96u);
static_assert(sizeof(ClipProjectionData) == 88u);

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
	virtual nbl::core::SRange<const PolylineConnector> getConnectors() const = 0;
	virtual bool checkSectionsContinuity() const = 0;
};

// It is not optimized because how you feed a Polyline to our cad renderer is your choice. this is just for convenience
// This is a Nabla Polyline used to feed to our CAD renderer. You can convert your Polyline to this class. or just use it directly.
class CPolyline : public CPolylineBase
{
public:
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

	nbl::core::SRange<const PolylineConnector> getConnectors() const override
	{
		return nbl::core::SRange<const PolylineConnector> { m_polylineConnector.begin()._Ptr, m_polylineConnector.end()._Ptr };
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

	void addLinePoints(const nbl::core::SRange<float64_t2>& linePoints, bool forceConnectToLastSection = false)
	{
		if (linePoints.size() <= 1u)
			return;

		SectionInfo newSection = {};
		newSection.type = ObjectType::LINE;
		newSection.index = static_cast<uint32_t>(m_linePoints.size());
		newSection.count = static_cast<uint32_t>(linePoints.size() - 1u);
		m_sections.push_back(newSection);

		const uint32_t oldLinePointSize = m_linePoints.size();
		const uint32_t newLinePointSize = oldLinePointSize + linePoints.size();
		m_linePoints.resize(newLinePointSize);
		for (uint32_t i = 0u; i < linePoints.size(); i++)
		{
			m_linePoints[oldLinePointSize + i].p = linePoints[i];
		}

		if (forceConnectToLastSection && m_sections.size() >= 2u)
		{
			float64_t2 prevPoint = getSectionLastPoint(m_sections[m_sections.size() - 2u]); // - 2 because we just added a new section
			m_linePoints[oldLinePointSize].p = prevPoint;
		}
	}

	void addEllipticalArcs(const nbl::core::SRange<curves::EllipticalArcInfo>& ellipses, double errorThreshold)
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

	// TODO[Przemek]: this input should be nbl::hlsl::QuadraticBezier instead cause `QuadraticBezierInfo` includes precomputed data I don't want user to see
	void addQuadBeziers(const nbl::core::SRange<nbl::hlsl::shapes::QuadraticBezier<double>>& quadBeziers, bool forceConnectToLastSection = false)
	{
		if (quadBeziers.empty())
			return;

		SectionInfo newSection = {};
		newSection.type = ObjectType::QUAD_BEZIER;
		newSection.index = static_cast<uint32_t>(m_quadBeziers.size());
		newSection.count = static_cast<uint32_t>(quadBeziers.size());
		m_sections.push_back(newSection);


		constexpr QuadraticBezierInfo EMPTY_QUADRATIC_BEZIER_INFO = {};
		const uint32_t oldQuadBezierSize = m_quadBeziers.size();
		const uint32_t newQuadBezierSize = oldQuadBezierSize + quadBeziers.size();
		m_quadBeziers.resize(newQuadBezierSize);
		for (uint32_t i = 0u; i < quadBeziers.size(); i++)
		{
			const uint32_t currBezierIdx = oldQuadBezierSize + i;
			m_quadBeziers[currBezierIdx].shape = quadBeziers[i];
		}

		if (forceConnectToLastSection && m_sections.size() >= 2u)
		{
			float64_t2 prevPoint = getSectionLastPoint(m_sections[m_sections.size() - 2u]); // - 2 because we just added a new section
			m_quadBeziers[oldQuadBezierSize].shape.P0 = prevPoint; // or we can average?
		}
	}

	void preprocessPolylineWithStyle(const CPULineStyle& lineStyle)
	{
		if (!lineStyle.isVisible())
			return;
		// DISCONNECTION DETECTED, will break styling and offsetting the polyline, if you don't care about those then ignore discontinuity.
		_NBL_DEBUG_BREAK_IF(!checkSectionsContinuity());
		PolylineConnectorBuilder connectorBuilder;

		float phaseShiftTotal = lineStyle.phaseShift;
		for (uint32_t sectionIdx = 0u; sectionIdx < m_sections.size(); sectionIdx++)
		{
			const auto& section = m_sections[sectionIdx];

			if (section.type == ObjectType::LINE)
			{
				// calculate phase shift at each point of each line in section
				const uint32_t linePointCnt = section.count + 1u;
				for (uint32_t i = 0u; i < linePointCnt; i++)
				{
					const uint32_t currIdx = section.index + i;
					auto& linePoint = m_linePoints[currIdx];
					if (i == 0u)
					{
						linePoint.phaseShift = phaseShiftTotal;
						continue;
					}

					const auto& prevLinePoint = m_linePoints[section.index + i - 1u];
					const float64_t2 lineVector = linePoint.p - prevLinePoint.p;
					const double lineLen = glm::length(lineVector);

					if (lineStyle.isRoadStyleFlag)
					{
						connectorBuilder.addLineNormal(lineVector, lineLen, prevLinePoint.p, phaseShiftTotal);
					}

					const double changeInPhaseShiftBetweenCurrAndPrevPoint = std::remainder(lineLen, 1.0f / lineStyle.reciprocalStipplePatternLen) * lineStyle.reciprocalStipplePatternLen;
					linePoint.phaseShift = static_cast<float32_t>(glm::fract(phaseShiftTotal + changeInPhaseShiftBetweenCurrAndPrevPoint));
					phaseShiftTotal = linePoint.phaseShift;
				}
			}
			else if (section.type == ObjectType::QUAD_BEZIER)
			{
				// calculate phase shift at point P0 of each bezier
				const uint32_t quadBezierCnt = section.count;
				for (uint32_t i = 0u; i <= quadBezierCnt; i++)
				{

					const uint32_t currIdx = section.index + i;
					if (i == 0u)
					{
						QuadraticBezierInfo& firstInSectionQuadBezierInfo = m_quadBeziers[currIdx];
						firstInSectionQuadBezierInfo.phaseShift = phaseShiftTotal;

						if (lineStyle.isRoadStyleFlag)
						{
							connectorBuilder.addBezierNormals(m_quadBeziers[currIdx], phaseShiftTotal);
						}

						continue;
					}

					const QuadraticBezierInfo& prevQuadBezierInfo = m_quadBeziers[currIdx - 1u];
					nbl::hlsl::shapes::Quadratic<double> quadratic = nbl::hlsl::shapes::Quadratic<double>::constructFromBezier(prevQuadBezierInfo.shape);
					nbl::hlsl::shapes::Quadratic<double>::ArcLengthCalculator arcLenCalc = nbl::hlsl::shapes::Quadratic<double>::ArcLengthCalculator::construct(quadratic);
					const double bezierLen = arcLenCalc.calcArcLen(1.0f);

					const double nextLineInSectionLocalPhaseShift = std::remainder(bezierLen, 1.0f / lineStyle.reciprocalStipplePatternLen) * lineStyle.reciprocalStipplePatternLen;
					phaseShiftTotal = static_cast<float32_t>(glm::fract(phaseShiftTotal + nextLineInSectionLocalPhaseShift));

					if (i < quadBezierCnt)
					{
						QuadraticBezierInfo& quadBezierInfo = m_quadBeziers[currIdx];
						quadBezierInfo.phaseShift = phaseShiftTotal;

						if (lineStyle.isRoadStyleFlag)
						{
							connectorBuilder.addBezierNormals(m_quadBeziers[currIdx], phaseShiftTotal);
						}
					}
				}
			}
		}

		if (lineStyle.isRoadStyleFlag)
		{
			connectorBuilder.setPhaseShiftAtEndOfPolyline(phaseShiftTotal);
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
						float64_t2 intersection = nbl::hlsl::shapes::util::LineLineIntersection(prevSectionEndPos, prevTangent, nextSectionStartPos, nextTangent);

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
								parallelPolyline.insertLinePointsToSection(nextSectionIdx, 0u, nbl::core::SRange<float64_t2>(newLinePoints, newLinePoints + 2u));
							}
						}
						else if (nextSection.type == ObjectType::QUAD_BEZIER)
						{
							if (prevSection.type == ObjectType::LINE)
							{
								// Add Intersection + right Segment first line position to end of leftSegment
								float64_t2 newLinePoints[2u] = { intersection, nextSectionStartPos };
								parallelPolyline.insertLinePointsToSection(prevSectionIdx, prevSection.count + 1u, nbl::core::SRange<float64_t2>(newLinePoints, newLinePoints + 2u));
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
								parallelPolyline.insertLinePointsToSection(newSectionIdx, 0u, nbl::core::SRange<float64_t2>(newLinePoints, newLinePoints + 3u));
								previousLineSectionIdx = newSectionIdx;
							}
						}
					}
					else // Inward Needs Trim and Prune
					{
						const SectionIntersectResult sectionIntersectResult = parallelPolyline.intersectTwoSections(prevSection, nextSection);

						if (sectionIntersectResult.valid())
						{
							assert(sectionIntersectResult.prevObjIndex < prevSection.count);
							assert(sectionIntersectResult.nextObjIndex < nextSection.count);
							parallelPolyline.removeSectionObjectsFromIdxToEnd(prevSectionIdx, sectionIntersectResult.prevObjIndex + 1u);
							parallelPolyline.removeSectionObjectsFromBeginToIdx(nextSectionIdx, sectionIntersectResult.nextObjIndex);
							if (nextSection.type == ObjectType::LINE)
							{
								if (prevSection.type == ObjectType::LINE)
								{
									parallelPolyline.m_linePoints[prevSection.index + prevSection.count].p = sectionIntersectResult.intersection;
									parallelPolyline.m_linePoints[nextSection.index].p = sectionIntersectResult.intersection;
								}
								else if (prevSection.type == ObjectType::QUAD_BEZIER)
								{
									parallelPolyline.m_quadBeziers[prevSection.index + prevSection.count - 1u].shape.splitFromStart(sectionIntersectResult.prevT);
									parallelPolyline.m_linePoints[nextSection.index].p = sectionIntersectResult.intersection;
								}
							}
							else if (nextSection.type == ObjectType::QUAD_BEZIER)
							{
								if (prevSection.type == ObjectType::LINE)
								{
									parallelPolyline.m_linePoints[prevSection.index + prevSection.count].p = sectionIntersectResult.intersection;
									parallelPolyline.m_quadBeziers[nextSection.index].shape.splitToEnd(sectionIntersectResult.nextT);
								}
								else if (prevSection.type == ObjectType::QUAD_BEZIER)
								{
									// TODO clip prev section last bezier from 0 to t0
									// TODO clip next section first bezier from t1 to 1.0

								}
							}
						}
						else
						{
							// TODO:
						}
					}
				}
			};
		auto connectBezierSection = [&](std::vector<nbl::hlsl::shapes::QuadraticBezier<double>>&& beziers)
			{
				parallelPolyline.addQuadBeziers(nbl::core::SRange<nbl::hlsl::shapes::QuadraticBezier<double>>(beziers.begin()._Ptr, beziers.end()._Ptr));
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
				parallelPolyline.addLinePoints(nbl::core::SRange<float64_t2>(linePoints.begin()._Ptr, linePoints.end()._Ptr));
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
						const float64_t2 tangent = glm::normalize(m_linePoints[linePointIdx + 1].p - m_linePoints[linePointIdx].p);
						offsetVector = float64_t2(tangent.y, -tangent.x);
					}
					else if (j == section.count)
					{
						const float64_t2 tangent = glm::normalize(m_linePoints[linePointIdx].p - m_linePoints[linePointIdx - 1].p);
						offsetVector = float64_t2(tangent.y, -tangent.x);
					}
					else
					{
						const float64_t2 tangentPrevLine = glm::normalize(m_linePoints[linePointIdx].p - m_linePoints[linePointIdx - 1].p);
						const float64_t2 normalPrevLine = float64_t2(tangentPrevLine.y, -tangentPrevLine.x);
						const float64_t2 tangentNextLine = glm::normalize(m_linePoints[linePointIdx + 1].p - m_linePoints[linePointIdx].p);
						const float64_t2 normalNextLine = float64_t2(tangentNextLine.y, -tangentNextLine.x);

						const float64_t2 intersectionDirection = glm::normalize(normalPrevLine + normalNextLine);
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

	void setClosed(bool closed)
	{
		m_closedPolygon = closed;
	}

protected:
	std::vector<PolylineConnector> m_polylineConnector;
	std::vector<SectionInfo> m_sections;
	std::vector<LinePointInfo> m_linePoints;
	std::vector<QuadraticBezierInfo> m_quadBeziers;

	// Next 3 are protected member functions to modify current lines and bezier sections used in polyline offsetting:

	void insertLinePointsToSection(uint32_t sectionIdx, uint32_t insertionPoint, const nbl::core::SRange<float64_t2>& linePoints)
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

		if (section.count > removedCount)
			section.count -= removedCount;
		else
			m_sections.erase(m_sections.begin() + sectionIdx);

		// update next sections offsets
		for (uint32_t i = sectionIdx + 1u; i < m_sections.size(); ++i)
		{
			if (m_sections[i].type == section.type)
				m_sections[i].index -= removedCount;
		}
	}

	void removeSectionObjectsFromBeginToIdx(uint32_t sectionIdx, uint32_t idx)
	{
		SectionInfo& section = m_sections[sectionIdx];
		if (idx <= 0)
			return;
		const size_t removedCount = idx;
		if (section.type == ObjectType::LINE)
		{
			if (idx >= section.count) // if idx==section.count it means it wants to delete the whole line section, so we remove all the line points including the last one in the section
				m_linePoints.erase(m_linePoints.begin() + section.index, m_linePoints.begin() + section.index + section.count + 1u);
			else
				m_linePoints.erase(m_linePoints.begin() + section.index, m_linePoints.begin() + section.index + idx);
		}
		else if (section.type == ObjectType::QUAD_BEZIER)
			m_quadBeziers.erase(m_quadBeziers.begin() + section.index, m_quadBeziers.begin() + section.index + idx);

		if (section.count > removedCount)
			section.count -= removedCount;
		else
			m_sections.erase(m_sections.begin() + sectionIdx);

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

			const float64_t2 intersection = nbl::hlsl::shapes::util::LineLineIntersection(A0, V0, A1, V1);
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

	// TODO: move this function to a better place, shared functionality with hatches
	static void BezierLineIntersection(nbl::hlsl::shapes::QuadraticBezier<float64_t> bezier, const float64_t2 lineStart, const float64_t2 lineVector, float64_t2& outBezierTs)
	{
		float64_t2 lineDir = glm::normalize(lineVector);
		float64_t2x2 rotate = float64_t2x2({ lineDir.x, lineDir.y }, { -lineDir.y, lineDir.x });
		bezier.P0 = mul(rotate, bezier.P0 - lineStart);
		bezier.P1 = mul(rotate, bezier.P1 - lineStart);
		bezier.P2 = mul(rotate, bezier.P2 - lineStart);
		nbl::hlsl::shapes::Quadratic<double> quadratic = nbl::hlsl::shapes::Quadratic<double>::constructFromBezier(bezier);
		outBezierTs = nbl::hlsl::math::equations::Quadratic<float64_t>::construct(quadratic.A.y, quadratic.B.y, quadratic.C.y).computeRoots();
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

			float64_t2 bezierTs;
			BezierLineIntersection(bezier, A, V, bezierTs);

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
		}

		return res;
	}

	SectionIntersectResult intersectTwoSections(const SectionInfo& prevSection, const SectionInfo& nextSection) const
	{
		SectionIntersectResult ret = {};
		ret.invalidate();

		if (prevSection.count == 0 || nextSection.count == 0)
			return ret;

		// tmp for testing
		if (prevSection.type == ObjectType::QUAD_BEZIER && nextSection.type == ObjectType::QUAD_BEZIER)
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

						if (prevSection.type == ObjectType::LINE && nextSection.type == ObjectType::LINE)
						{
							SectionIntersectResult localIntersectionResult = intersectLineSectionObjects(prevSection, prevObjIdx, nextSection, nextObjIdx);
							if (localIntersectionResult.valid())
							{
								// TODO: Better Criterial to select between multiple intersections of the same objects
								if (objectsToRemove > currentIntersectionObjectRemoveCount)
								{
									ret = localIntersectionResult;
									currentIntersectionObjectRemoveCount = objectsToRemove;
								}
							}
						}
						else if ((prevSection.type == ObjectType::QUAD_BEZIER && nextSection.type == ObjectType::LINE) || (prevSection.type == ObjectType::LINE && nextSection.type == ObjectType::QUAD_BEZIER))
						{
							std::array<SectionIntersectResult, 2> localIntersectionResults = intersectLineBezierSectionObjects(prevSection, prevObjIdx, nextSection, nextObjIdx);
							for (const auto& localIntersectionResult : localIntersectionResults)
							{
								if (localIntersectionResult.valid())
								{
									// TODO: Better Criterial to select between multiple intersections of the same objects
									if (objectsToRemove > currentIntersectionObjectRemoveCount)
									{
										ret = localIntersectionResult;
										currentIntersectionObjectRemoveCount = objectsToRemove;
									}
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

		std::vector<PolylineConnector> buildConnectors(const CPULineStyle& lineStyle, bool isClosedPolygon)
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

		bool checkIfInDrawSection(const CPULineStyle& lineStyle, float normalizedPlaceInPattern)
		{
			float integralPart;
			normalizedPlaceInPattern = std::modf(normalizedPlaceInPattern, &integralPart);
			const float* patternPtr = std::upper_bound(lineStyle.stipplePattern, lineStyle.stipplePattern + lineStyle.stipplePatternSize, normalizedPlaceInPattern);
			const uint32_t patternIdx = std::distance(lineStyle.stipplePattern, patternPtr);
			// odd patternIdx means a "no draw section" and current candidate should split into two nearest draw sections
			return !(patternIdx & 0x1);
		}

		void constructMiterIfVisible(
			const CPULineStyle& lineStyle,
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

	// important for miter and parallel generation
	bool m_closedPolygon = false;
};
