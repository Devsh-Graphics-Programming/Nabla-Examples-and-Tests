#pragma once

#include <nabla.h>
#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/math/geometry.hlsl>
#include <nbl/builtin/hlsl/shapes/util.hlsl>
#include "curves.h"

using namespace nbl;
using namespace ui;

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
			phaseShift -= 1e-3; // TODO: I think 1e-3 phase shift in normalized stipple space is a reasonable value? right?
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

	size_t getSectionsCount() const { return m_sections.size(); }

	const SectionInfo& getSectionInfoAt(const uint32_t idx) const
	{
		return m_sections[idx];
	}

	const QuadraticBezierInfo& getQuadBezierInfoAt(const uint32_t idx) const
	{
		return m_quadBeziers[idx];
	}

	const LinePointInfo& getLinePointAt(const uint32_t idx) const
	{
		return m_linePoints[idx];
	}

	const std::vector<PolylineConnector>& getConnectors() const
	{
		return m_polylineConnector;
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

	void addLinePoints(const core::SRange<float64_t2>& linePoints, bool addToPreviousLineSectionIfAvailable = false)
	{
		if (linePoints.size() <= 1u)
			return;

		const bool previousSectionIsLine = m_sections.size() > 0u && m_sections[m_sections.size() - 1u].type == ObjectType::LINE;
		const bool alwaysAddNewSection = !addToPreviousLineSectionIfAvailable;
		const bool addNewSection = alwaysAddNewSection || !previousSectionIsLine;
		if (addNewSection)
		{
			SectionInfo newSection = {};
			newSection.type = ObjectType::LINE;
			newSection.index = static_cast<uint32_t>(m_linePoints.size());
			newSection.count = static_cast<uint32_t>(linePoints.size() - 1u);
			m_sections.push_back(newSection);
		}
		else
		{
			m_sections[m_sections.size() - 1u].count += static_cast<uint32_t>(linePoints.size());
		}

		const uint32_t oldLinePointSize = m_linePoints.size();
		const uint32_t newLinePointSize = oldLinePointSize + linePoints.size();
		m_linePoints.resize(newLinePointSize);
		for (uint32_t i = 0u; i < linePoints.size(); i++)
		{
			m_linePoints[oldLinePointSize + i].p = linePoints[i];
		}
	}

	void addEllipticalArcs(const core::SRange<curves::EllipticalArcInfo>& ellipses)
	{
		// TODO[Erfan] Approximate with quadratic beziers
	}

	// TODO[Przemek]: this input should be nbl::hlsl::QuadraticBezier instead cause `QuadraticBezierInfo` includes precomputed data I don't want user to see
	void addQuadBeziers(const core::SRange<shapes::QuadraticBezier<double>>& quadBeziers, bool addToPreviousSectionIfAvailable = false)
	{
		const bool previousSectionIsBezier = m_sections.size() > 0u && m_sections[m_sections.size() - 1u].type == ObjectType::QUAD_BEZIER;
		const bool alwaysAddNewSection = !addToPreviousSectionIfAvailable;
		bool addNewSection = alwaysAddNewSection || !previousSectionIsBezier;
		if (addNewSection)
		{
			SectionInfo newSection = {};
			newSection.type = ObjectType::QUAD_BEZIER;
			newSection.index = static_cast<uint32_t>(m_quadBeziers.size());
			newSection.count = static_cast<uint32_t>(quadBeziers.size());
			m_sections.push_back(newSection);
		}
		else
		{
			m_sections[m_sections.size() - 1u].count += static_cast<uint32_t>(quadBeziers.size());
		}


		constexpr QuadraticBezierInfo EMPTY_QUADRATIC_BEZIER_INFO = {};
		const uint32_t oldQuadBezierSize = m_quadBeziers.size();
		const uint32_t newQuadBezierSize = oldQuadBezierSize + quadBeziers.size();
		m_quadBeziers.resize(newQuadBezierSize);
		for (uint32_t i = 0u; i < quadBeziers.size(); i++)
		{
			const uint32_t currBezierIdx = oldQuadBezierSize + i;
			m_quadBeziers[currBezierIdx].p[0] = quadBeziers[i].P0;
			m_quadBeziers[currBezierIdx].p[1] = quadBeziers[i].P1;
			m_quadBeziers[currBezierIdx].p[2] = quadBeziers[i].P2;
		}
	}

	void preprocessPolylineWithStyle(const CPULineStyle& lineStyle)
	{
		// DISCONNECTION DETECTED, will break styling and offsetting the polyline, if you don't care about those then ignore discontinuity.
		_NBL_DEBUG_BREAK_IF(!checkSectionsContunuity());
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
					const float64_t2 line = linePoint.p - prevLinePoint.p;
					const double lineLen = glm::length(line);

					if (lineStyle.isRoadStyleFlag)
					{
						connectorBuilder.addLineNormal(line, lineLen, prevLinePoint.p, phaseShiftTotal);
					}

					const double changeInPhaseShiftBetweenCurrAndPrevPoint = std::remainder(lineLen, 1.0f / lineStyle.reciprocalStipplePatternLen) * lineStyle.reciprocalStipplePatternLen;
					linePoint.phaseShift = glm::fract(phaseShiftTotal + changeInPhaseShiftBetweenCurrAndPrevPoint);
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
					shapes::QuadraticBezier<double> quadraticBezier = shapes::QuadraticBezier<double>::construct(prevQuadBezierInfo.p[0], prevQuadBezierInfo.p[1], prevQuadBezierInfo.p[2]);
					shapes::Quadratic<double> quadratic = shapes::Quadratic<double>::constructFromBezier(quadraticBezier);
					shapes::Quadratic<double>::ArcLengthCalculator arcLenCalc = shapes::Quadratic<double>::ArcLengthCalculator::construct(quadratic);
					const double bezierLen = arcLenCalc.calcArcLen(1.0f);

					const double nextLineInSectionLocalPhaseShift = std::remainder(bezierLen, 1.0f / lineStyle.reciprocalStipplePatternLen) * lineStyle.reciprocalStipplePatternLen;
					phaseShiftTotal = glm::fract(phaseShiftTotal + nextLineInSectionLocalPhaseShift);

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
			return m_quadBeziers[firstBezierIdx].p[0];
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
			return m_quadBeziers[lastBezierIdx].p[2];
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
			return m_quadBeziers[firstBezierIdx].p[1] - m_quadBeziers[firstBezierIdx].p[0];
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
			return m_quadBeziers[lastBezierIdx].p[2] - m_quadBeziers[lastBezierIdx].p[1];
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
		_NBL_DEBUG_BREAK_IF(!checkSectionsContunuity());

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
					float64_t2 prevSectionEndPos = parallelPolyline.getSectionLastPoint(prevSection);
					float64_t2 nextSectionStartPos = parallelPolyline.getSectionFirstPoint(nextSection);
					float64_t2 intersection = nbl::hlsl::shapes::util::LineLineIntersection(prevSectionEndPos, prevTangent, nextSectionStartPos, nextTangent);

					if (crossProduct * offset > 0u) // Outward, needs connection
					{
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
								parallelPolyline.insertLinePointsToSection(nextSectionIdx, 0u, core::SRange<float64_t2>(newLinePoints, newLinePoints + 2u));
							}
						}
						else if (nextSection.type == ObjectType::QUAD_BEZIER)
						{
							if (prevSection.type == ObjectType::LINE)
							{
								// Add Intersection + right Segment first line position to end of leftSegment
								float64_t2 newLinePoints[2u] = { intersection, nextSectionStartPos };
								parallelPolyline.insertLinePointsToSection(prevSectionIdx, prevSection.count + 1u, core::SRange<float64_t2>(newLinePoints, newLinePoints + 2u));
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
								parallelPolyline.insertLinePointsToSection(newSectionIdx, 0u, core::SRange<float64_t2>(newLinePoints, newLinePoints + 3u));
								previousLineSectionIdx = newSectionIdx;
							}
						}
					}
					else // Inward Needs Trim and Prune
					{
						if (nextSection.type == ObjectType::LINE)
						{
							if (prevSection.type == ObjectType::LINE)
							{
								uint32_t prevIdx = prevSection.count - 1u;
								uint32_t nextIdx = 0u;

								// TODO: Do stuff to find these values

								const bool prevSectionModified = prevIdx < prevSection.count - 1u;
								const bool nextSectionModified = nextIdx > 0;

								parallelPolyline.removeSectionObjectsFromIdxToEnd(prevSectionIdx, prevIdx +1u);
								parallelPolyline.removeSectionObjectsFromBeginToIdx(nextSectionIdx, nextIdx);

								// intersect again if prev and next section have changed
								if (prevSectionModified)
								{
									prevTangent = parallelPolyline.getSectionLastTangent(prevSection);
									prevSectionEndPos = parallelPolyline.getSectionLastPoint(prevSection);
								}
								if (nextSectionModified)
								{
									nextTangent = parallelPolyline.getSectionFirstTangent(nextSection);
									nextSectionStartPos = parallelPolyline.getSectionFirstPoint(nextSection);
								}
								if (prevSectionModified || nextSectionModified)
									intersection = nbl::hlsl::shapes::util::LineLineIntersection(prevSectionEndPos, prevTangent, nextSectionStartPos, nextTangent);

								parallelPolyline.m_linePoints[prevSection.index + prevSection.count].p = intersection;
								parallelPolyline.m_linePoints[nextSection.index].p = intersection;
							}
							else if (prevSection.type == ObjectType::QUAD_BEZIER)
							{
								uint32_t prevIdx = prevSection.count - 1u;
								uint32_t nextIdx = 0u;
								float64_t t = 1.0; // for intersected bezier

								// TODO: Do stuff to find these values

								parallelPolyline.removeSectionObjectsFromIdxToEnd(prevSectionIdx, prevIdx + 1u);
								parallelPolyline.removeSectionObjectsFromBeginToIdx(nextSectionIdx, nextIdx);

								// TODO clip prev section last bezier from 0 to t0
								// TODO Set next section first point to bezier eval at 1.0
							}
						}
						else if (nextSection.type == ObjectType::QUAD_BEZIER)
						{
							if (prevSection.type == ObjectType::LINE)
							{
								uint32_t prevIdx = prevSection.count - 1u;
								uint32_t nextIdx = 0u;
								float64_t t = 0.0; // for intersected bezier

								// TODO: Do stuff to find these values

								parallelPolyline.removeSectionObjectsFromIdxToEnd(prevSectionIdx, prevIdx + 1u);
								parallelPolyline.removeSectionObjectsFromBeginToIdx(nextSectionIdx, nextIdx);

								// TODO Set prev section last point to bezier eval at 0.0
								// TODO clip next section first bezier from t1 to 1.0
							}
							else if (prevSection.type == ObjectType::QUAD_BEZIER)
							{
								uint32_t prevIdx = prevSection.count - 1u;
								uint32_t nextIdx = 0u;
								float64_t t0 = 1.0; // for prev intersected bezier
								float64_t t1 = 0.0; // for next intersected bezier

								// TODO: Do stuff to find these values

								parallelPolyline.removeSectionObjectsFromIdxToEnd(prevSectionIdx, prevIdx + 1u);
								parallelPolyline.removeSectionObjectsFromBeginToIdx(nextSectionIdx, nextIdx);

								// TODO clip prev section last bezier from 0 to t0
								// TODO clip next section first bezier from t1 to 1.0

							}
						}
					}
				}
			};
		auto connectBezierSection = [&](std::vector<shapes::QuadraticBezier<double>>&& beziers)
			{
				parallelPolyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<double>>(beziers.begin()._Ptr, beziers.end()._Ptr));
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
				parallelPolyline.addLinePoints(core::SRange<float64_t2>(linePoints.begin()._Ptr, linePoints.end()._Ptr));
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
				std::vector<shapes::QuadraticBezier<double>> newBeziers;
				curves::Subdivision::AddBezierFunc addToBezier = [&](shapes::QuadraticBezier<double>&& info) -> void
					{
						newBeziers.push_back(info);
					};
				for (uint32_t j = 0; j < section.count; ++j)
				{
					const uint32_t bezierIdx = section.index + j;
					const shapes::QuadraticBezier<double>& bezier = shapes::QuadraticBezier<double>::construct(m_quadBeziers[bezierIdx].p[0], m_quadBeziers[bezierIdx].p[1], m_quadBeziers[bezierIdx].p[2]);
					curves::OffsettedBezier offsettedBezier(bezier, offset);
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

	bool checkSectionsContunuity() const
	{
		// Check for continuity
		for (uint32_t i = 1; i < m_sections.size(); ++i)
		{
			constexpr float64_t POINT_EQUALITY_THRESHOLD = 1e-12;
			const float64_t2 firstPoint = getSectionFirstPoint(m_sections[i]);
			const float64_t2 prevLastPoint = getSectionLastPoint(m_sections[i-1u]);
			if (glm::distance(firstPoint, prevLastPoint) > POINT_EQUALITY_THRESHOLD)
			{
				// DISCONNECTION DETECTED, will break styling and offsetting the polyline, if you don't care about those then ignore discontinuity.
				return false;
			}
		}
		return true;
	}

protected:
	std::vector<PolylineConnector> m_polylineConnector;
	std::vector<SectionInfo> m_sections;
	std::vector<LinePointInfo> m_linePoints;
	std::vector<QuadraticBezierInfo> m_quadBeziers;

	// Next 3 are protected member functions to modify current lines and bezier sections used in polyline offsetting:
	
	void insertLinePointsToSection(uint32_t sectionIdx, uint32_t insertionPoint, const core::SRange<float64_t2>& linePoints)
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

private:
	class PolylineConnectorBuilder
	{
	public:
		void addLineNormal(const float64_t2& line, float lineLen, const float64_t2& worldSpaceCircleCenter, float phaseShift)
		{
			PolylineConnectorNormalHelperInfo connectorNormalInfo;
			connectorNormalInfo.worldSpaceCircleCenter = worldSpaceCircleCenter;
			connectorNormalInfo.normal = float32_t2(-line.y, line.x) / static_cast<float>(lineLen);
			connectorNormalInfo.phaseShift = phaseShift;
			connectorNormalInfo.type = ObjectType::LINE;

			connectorNormalInfos.push_back(connectorNormalInfo);
		}

		void addBezierNormals(const QuadraticBezierInfo& quadBezierInfo, float phaseShift)
		{
			// TODO: we already calculate quadratic form of each bezier (except of the last one), maybe store this info in an array and use it later to calculate normals?
			const float64_t2 bezierDerivativeValueAtP0 = 2.0 * (quadBezierInfo.p[1] - quadBezierInfo.p[0]);
			const float32_t2 tangentAtP0 = glm::normalize(bezierDerivativeValueAtP0);
			//const float_t2 A = P0 - 2.0 * P1 + P2;
			const float64_t2 bezierDerivativeValueAtP2 = 2.0 * (quadBezierInfo.p[0] - 2.0 * quadBezierInfo.p[1] + quadBezierInfo.p[2]) + 2.0 * (quadBezierInfo.p[1] - quadBezierInfo.p[0]);
			const float32_t2 tangentAtP2 = glm::normalize(bezierDerivativeValueAtP2);

			PolylineConnectorNormalHelperInfo connectorNormalInfoAtP0{};
			connectorNormalInfoAtP0.worldSpaceCircleCenter = quadBezierInfo.p[0];
			connectorNormalInfoAtP0.normal = float32_t2(-tangentAtP0.y, tangentAtP0.x);
			connectorNormalInfoAtP0.phaseShift = phaseShift;
			connectorNormalInfoAtP0.type = ObjectType::QUAD_BEZIER;

			PolylineConnectorNormalHelperInfo connectorNormalInfoAtP2{};
			connectorNormalInfoAtP2.worldSpaceCircleCenter = quadBezierInfo.p[2];
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
			constexpr float CROSS_PRODUCT_LINEARITY_EPSILON = 1e-6;
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

		core::vector<PolylineConnectorNormalHelperInfo> connectorNormalInfos;
		float phaseShiftAtEndOfPolyline = 0.0f;
	};

	// important for miter and parallel generation
	bool m_closedPolygon = false;
};
