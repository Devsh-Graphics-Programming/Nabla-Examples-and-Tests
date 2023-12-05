#pragma once

#include <nabla.h>
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
		ret.phaseShift = phaseShift;

		return ret;
	}

	inline bool isVisible() const { return stipplePatternSize != InvalidStipplePatternSize; }
};

static_assert(sizeof(DrawObject) == 16u);
static_assert(sizeof(MainObject) == 8u);
static_assert(sizeof(Globals) == 112u);
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

	void addLinePoints(const core::SRange<float64_t2>& linePoints, bool addToPreviousLineSectionIfAvailable = false)
	{
		// TODO[Erfan]: add warning debug breaks when there is breaks between added beziers and lines
		if (linePoints.size() <= 1u)
			return;

		const bool previousSectionIsLine = m_sections.size() > 0u && m_sections[m_sections.size() - 1u].type == ObjectType::LINE;
		const bool alwaysAddNewSection = !addToPreviousLineSectionIfAvailable;
		bool addNewSection = alwaysAddNewSection || !previousSectionIsLine;
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
		m_linePoints.insert(m_linePoints.end(), linePoints.begin(), linePoints.end());
	}

	void addEllipticalArcs(const core::SRange<curves::EllipticalArcInfo>& ellipses)
	{
		// TODO[Erfan] Approximate with quadratic beziers
	}

	// TODO[Przemek]: this input should be nbl::hlsl::QuadraticBezier instead cause `QuadraticBezierInfo` includes precomputed data I don't want user to see
	void addQuadBeziers(const core::SRange<QuadraticBezierInfo>& quadBeziers, bool addToPreviousSectionIfAvailable = false)
	{
		// TODO[Erfan]: add warning debug breaks when there is breaks between added beziers and lines
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
		m_quadBeziers.insert(m_quadBeziers.end(), quadBeziers.begin(), quadBeziers.end());
	}

	// TODO[Przemek]: Add a function here named preprocessPolylineWithStyle -> give it the line style
	/*
	*  this preprocess should:
	*	1. if style has road info try to generate miters:
	*		if tangents are not in the same direction with some error add a PolylineConnector object
		2. go over the list of sections (line and beziers in order) compute the phase shift by computing their arclen and divisions with style length and
			fill the phaseShift part of the QuadraticBezierInfo and LinePointInfo,
			you initially set them to 0 in addLinePoints/addQuadBeziers

		NOTE that PolylineConnectors are special object types, user does not add them and they should not be part of m_sections vector
	*/

	CPolyline generateParallelPolyline(float64_t offset, const float64_t maxError = 1e-5) const 
	{
		CPolyline parallelPolyline = {};
		parallelPolyline.setClosed(m_closedPolygon);

		std::vector<float64_t2> newLinePoints;
		std::vector<QuadraticBezierInfo> newBeziers;
		curves::Subdivision::AddBezierFunc addToBezier = [&](QuadraticBezierInfo&& info) -> void
			{
				newBeziers.push_back(info);
			};

		// Generate Mitered Line Sections and Offseted Beziers -> will still have breaks and disconnections after this loop
		for (uint32_t i = 0; i < m_sections.size(); ++i)
		{
			const auto& section = m_sections[i];
			if (section.type == ObjectType::LINE)
			{
				newLinePoints.reserve(m_linePoints.size());
				for (uint32_t j = 0; j < section.count + 1; ++j)
				{
					const uint32_t linePointIdx = section.index + j;
					float64_t2 offsetVector;
					if (j == 0)
					{
						const float64_t2 tangent = glm::normalize(m_linePoints[linePointIdx + 1] - m_linePoints[linePointIdx]);
						offsetVector = float64_t2(tangent.y, -tangent.x);
					}
					else if (j == section.count)
					{
						const float64_t2 tangent = glm::normalize(m_linePoints[linePointIdx] - m_linePoints[linePointIdx - 1]);
						offsetVector = float64_t2(tangent.y, -tangent.x);
					}
					else
					{
						const float64_t2 tangentPrevLine = glm::normalize(m_linePoints[linePointIdx] - m_linePoints[linePointIdx - 1]);
						const float64_t2 normalPrevLine = float64_t2(tangentPrevLine.y, -tangentPrevLine.x);
						const float64_t2 tangentNextLine = glm::normalize(m_linePoints[linePointIdx + 1] - m_linePoints[linePointIdx]);
						const float64_t2 normalNextLine = float64_t2(tangentNextLine.y, -tangentNextLine.x);

						const float64_t2 intersectionDirection = glm::normalize(normalPrevLine + normalNextLine);
						const float64_t cosAngleBetweenNormals = glm::dot(normalPrevLine, normalNextLine);
						offsetVector = intersectionDirection * sqrt(2.0 / (1.0 + cosAngleBetweenNormals));
					}
					newLinePoints.push_back(m_linePoints[linePointIdx] + offsetVector * offset);
				}
				parallelPolyline.addLinePoints(nbl::core::SRange<float64_t2>(newLinePoints.begin()._Ptr, newLinePoints.end()._Ptr));
				newLinePoints.clear();
			}
			else if (section.type == ObjectType::QUAD_BEZIER)
			{
				for (uint32_t j = 0; j < section.count; ++j)
				{
					const uint32_t bezierIdx = section.index + j;
					// After Merge: const nbl::hlsl::shapes::QuadraticBezier bezier = from current bezier;
					const QuadraticBezierInfo& bezier = m_quadBeziers[bezierIdx];
					curves::OffsettedBezier offsettedBezier(bezier, offset);
					curves::Subdivision::adaptive(offsettedBezier, maxError, addToBezier, 10u);
				}
				parallelPolyline.addQuadBeziers(nbl::core::SRange<QuadraticBezierInfo>(newBeziers.begin()._Ptr, newBeziers.end()._Ptr));
				newBeziers.clear();
			}
		}

		// TODO: Remove and use the one in shapes/util.hlsl
		auto LineLineIntersection = [](const float64_t2& p1, const float64_t2& v1, const float64_t2& p2, const float64_t2& v2)
			{
				float64_t denominator = v1.y * v2.x - v1.x * v2.y;
				float64_t2 diff = p1 - p2;
				float64_t numerator = dot(float64_t2(v2.y, -v2.x), float64_t2(diff.x, diff.y));

				if (abs(denominator) < 1e-15 && abs(numerator) < 1e-15)
				{
					// are parallel and the same
					return (p1 + p2) / 2.0;
				}

				float64_t t = numerator / denominator;
				float64_t2 intersectionPoint = p1 + t * v1;
				return intersectionPoint;
			};

		// Join Sections Together -> We assume G1 continuity in a single section (the usage of how black box g1 continous curves are estimated as beziers and added as a section/chunk)
		for (uint32_t i = 0; i < parallelPolyline.m_sections.size(); ++i)
		{
			// iteration i, joins section i and i + 1, last section may need join based on the polyline being closed.
			if (i == parallelPolyline.m_sections.size() - 1u && !parallelPolyline.m_closedPolygon)
				break;

			auto& leftSegment = parallelPolyline.m_sections[i];
			auto& rightSegment = (i < parallelPolyline.m_sections.size() - 1u) ? parallelPolyline.m_sections[i+1] : parallelPolyline.m_sections[0];
			float64_t2 leftTangent;
			float64_t2 rightTangent;
			{
				if (leftSegment.type == ObjectType::LINE)
				{
					const uint32_t lastLinePointIdx = leftSegment.index + leftSegment.count;
					leftTangent = parallelPolyline.m_linePoints[lastLinePointIdx] - parallelPolyline.m_linePoints[lastLinePointIdx - 1];
				}
				else if (leftSegment.type == ObjectType::QUAD_BEZIER)
				{
					const uint32_t lastBezierIdx = leftSegment.index + leftSegment.count - 1u;
					leftTangent = parallelPolyline.m_quadBeziers[lastBezierIdx].p[2] - parallelPolyline.m_quadBeziers[lastBezierIdx].p[1];
				}

				if (rightSegment.type == ObjectType::LINE)
				{
					const uint32_t firstLinePointIdx = rightSegment.index;
					rightTangent = parallelPolyline.m_linePoints[firstLinePointIdx + 1u] - parallelPolyline.m_linePoints[firstLinePointIdx];
				}
				else if (rightSegment.type == ObjectType::QUAD_BEZIER)
				{
					const uint32_t firstBezierIdx = rightSegment.index;
					rightTangent = parallelPolyline.m_quadBeziers[firstBezierIdx].p[1] - parallelPolyline.m_quadBeziers[firstBezierIdx].p[0];
				}
			}

			// TODO: Replace with cross 2d
			constexpr float64_t CROSS_PRODUCT_LINEARITY_EPSILON = 1e-5;
			const float64_t crossProduct = leftTangent.x * rightTangent.y - leftTangent.y * rightTangent.x;

			int32_t lineSegmentsOffsetChange = 0;

			if (abs(crossProduct) > CROSS_PRODUCT_LINEARITY_EPSILON)
			{
				if (crossProduct * offset > 0u)
				{
					// outward, need to join by extra lines
					if (leftSegment.type == ObjectType::LINE)
					{
						const uint32_t lastLinePointIdx = leftSegment.index + leftSegment.count;
						float64_t2 lineEndPos = parallelPolyline.m_linePoints[lastLinePointIdx];
						if (rightSegment.type == ObjectType::QUAD_BEZIER)
						{
							const uint32_t firstBezierIdx = rightSegment.index;
							float64_t2 bezierStartPos = parallelPolyline.m_quadBeziers[firstBezierIdx].p[0];
							float64_t2 intersection = LineLineIntersection(lineEndPos, leftTangent, bezierStartPos, rightTangent);
							parallelPolyline.m_linePoints.insert(parallelPolyline.m_linePoints.begin() + lastLinePointIdx + 1u, intersection);
							parallelPolyline.m_linePoints.insert(parallelPolyline.m_linePoints.begin() + lastLinePointIdx + 1u, bezierStartPos);
							lineSegmentsOffsetChange += 2;
							// Add Intersection + QuadBez Position to end of leftSegment
						}
						else if (rightSegment.type == ObjectType::LINE)
						{
							// Add Intersection + right Segment first line position to end of leftSegment
						}
					}
					else if (leftSegment.type == ObjectType::QUAD_BEZIER)
					{
						if (rightSegment.type == ObjectType::QUAD_BEZIER)
						{
							// Add QuadBez Position + Intersection + QuadBez Position to new segment
						}
						else if (rightSegment.type == ObjectType::LINE)
						{
							// Add QuadBez Position + Intersection to start of right segment
						}
					}
				}
				else
				{
					// intersection goes inward, need to prune and clip
				}
			}
		}

		return parallelPolyline;
	}

	void setClosed(bool closed)
	{
		m_closedPolygon = closed;
	}

protected:
	// TODO[Przemek]: a vector of polyline connetor objects
	std::vector<SectionInfo> m_sections;
	// TODO[Przemek]: instead of float64_t2 for linePoints, store LinePointInfo
	std::vector<float64_t2> m_linePoints;
	std::vector<QuadraticBezierInfo> m_quadBeziers;

	// important for miter and parallel generation
	bool m_closedPolygon = false;
};
