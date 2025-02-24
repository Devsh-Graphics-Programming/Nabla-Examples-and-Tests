#include "Polyline.h"

void CPolyline::preprocessPolylineWithStyle(const LineStyleInfo& lineStyle, float64_t discontinuityErrorTolerance, const AddShapeFunc& addShape)
{
	if (lineStyle.skipPreprocess())
		return;
	const float64_t2 DiscontinuityErrorTolerance = float64_t2(discontinuityErrorTolerance, discontinuityErrorTolerance);
	// We allow for discontinuity now, so no need to enable this unless testing.
	// DISCONNECTION DETECTED, will break styling and offsetting the polyline, if you don't care about those then ignore discontinuity.
	// _NBL_DEBUG_BREAK_IF(!checkSectionsContinuity());

	// Check if it's truly closedPolygon
	bool actuallyClosed = false;
	if (m_closedPolygon && m_sections.size() > 0u)
		actuallyClosed = checkSectionsActuallyClosed(discontinuityErrorTolerance);

	const bool shouldAddShapes = (lineStyle.hasShape() && addShape.operator bool());
	// When stretchToFit is true, the curve section and individual lines should start from the beginning of the pattern (phaseShift = lineStyle.phaseShift)
	float currentPhaseShift = lineStyle.phaseShift;

	m_polylineConnector.clear();
	// to detect gap/discontinuity and reset phase shift
	float64_t2 prevPoint = float64_t2(nbl::hlsl::numeric_limits<float64_t>::infinity, nbl::hlsl::numeric_limits<float64_t>::infinity);
	static constexpr float64_t2 InvalidNormal = float64_t2(nbl::hlsl::numeric_limits<float64_t>::infinity, nbl::hlsl::numeric_limits<float64_t>::infinity);
	float64_t2 prevNormal = InvalidNormal;

	for (uint32_t sectionIdx = 0u; sectionIdx < m_sections.size(); sectionIdx++)
	{
		const auto& section = m_sections[sectionIdx];

		if (section.count == 0u)
		{
			assert(false); // shouldn't happen in any scenario
			continue;
		}

		// if gap detected
		if (sectionIdx > 0u && glm::any(glm::greaterThan(glm::abs(getSectionFirstPoint(section) - prevPoint), DiscontinuityErrorTolerance)))
		{
			if constexpr (PolylineSettings::ResetLineStyleOnDiscontinuity)
				currentPhaseShift = lineStyle.phaseShift; // reset phase shift
			prevNormal = InvalidNormal; // to avoid construction of polyline connector after a gap
		}

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

				if (lineStyle.stretchToFit)
					currentPhaseShift = lineStyle.getStretchedPhaseShift(stretchValue);

				linePoint.phaseShift = currentPhaseShift;
				linePoint.stretchValue = stretchValue;

				if (lineStyle.isRoadStyleFlag)
				{
					float64_t2 lineNormal = float64_t2(-lineVector.y, lineVector.x) / lineLen;
					if (prevNormal != InvalidNormal && checkIfInDrawSection(lineStyle, currentPhaseShift))
						addMiterIfVisible(prevNormal, lineNormal, linePoint.p);
					prevNormal = lineNormal;
				}

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

				if (!lineStyle.stretchToFit)
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

			if (lineStyle.stretchToFit)
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
				{
					const float32_t2 tangentAtP0 = glm::normalize(quadBezierInfo.shape.derivative(0.0));
					const float32_t2 tangentAtP2 = glm::normalize(quadBezierInfo.shape.derivative(1.0));
					const float64_t2 normalAtP0 = float32_t2(-tangentAtP0.y, tangentAtP0.x);
					const float64_t2 normalAtP2 = float32_t2(-tangentAtP2.y, tangentAtP2.x);
					if (prevNormal != InvalidNormal && checkIfInDrawSection(lineStyle, currentPhaseShift))
						addMiterIfVisible(prevNormal, normalAtP0, quadBezierInfo.shape.P0);
					prevNormal = normalAtP2;
				}

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
						float64_t t = arcLenCalc.calcArcLenInverse(quadratic, 0.0, 1.0, currentWorldSpaceOffset, 1e-5, 0.5); // todo: use discontinuityErrorTolerance here instead of 1e-5? same order? 
						addShape(quadratic.evaluate(t), quadratic.derivative(t), stretchValue);
						currentWorldSpaceOffset += stretchedPatternLen;
					}
				}

				const double changeInPhaseShift = glm::fract(bezierLen * rcpStretchedPatternLen);
				currentPhaseShift = static_cast<float32_t>(glm::fract(currentPhaseShift + changeInPhaseShift));
			}
		}

		prevPoint = getSectionLastPoint(section);
	}

	if (lineStyle.isRoadStyleFlag && actuallyClosed)
	{
		const auto& firstSection = m_sections.front();

		const float32_t2 firstTangent = glm::normalize(getSectionFirstTangent(firstSection));
		const float64_t2 firstNormal = float32_t2(-firstTangent.y, firstTangent.x);

		if (checkIfInDrawSection(lineStyle, lineStyle.phaseShift) && checkIfInDrawSection(lineStyle, currentPhaseShift))
			addMiterIfVisible(prevNormal, firstNormal, prevPoint);
	}
}

// outputs two offsets to the polyline and connects the ends if not closed
CPolyline CPolyline::generateParallelPolyline(float64_t offset, const float64_t maxError) const
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

void CPolyline::makeWideWhole(CPolyline& outOffset1, CPolyline& outOffset2, float64_t offset, const float64_t maxError) const
{
	outOffset1 = generateParallelPolyline(offset, maxError);
	outOffset2 = generateParallelPolyline(-1.0 * offset, maxError);

	if (!m_closedPolygon)
	{
		if (outOffset1.getSectionsCount() == 0u || outOffset2.getSectionsCount() == 0u)
			return;

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

void CPolyline::stippleBreakDown(const LineStyleInfo& lineStyle, const OutputPolylineFunc& addPolyline, float64_t discontinuityErrorTolerance) const
{
	if (!lineStyle.isVisible())
		return;

	// currently only works for road styles with only 2 stipple values (1 draw, 1 gap)
	assert(lineStyle.stipplePatternSize <= 1);

	const float64_t patternLen = 1.0 / lineStyle.reciprocalStipplePatternLen;
	const float32_t drawSectionNormalizedLen = lineStyle.stipplePattern[0];
	const float32_t gapSectionNormalizedLen = 1.0 - lineStyle.stipplePattern[0];
	const float32_t drawSectionLen = drawSectionNormalizedLen * patternLen;
	const bool allSolid = drawSectionNormalizedLen == 1.0f;
	const bool continous = checkSectionsContinuity(discontinuityErrorTolerance);

	if (allSolid && continous)
	{
		addPolyline(*this); // optimization to avoid copying and processing each individual line and bezier again.
		return;
	}
	
	// To detect gaps and flush
	const float64_t2 DiscontinuityErrorTolerance = float64_t2(discontinuityErrorTolerance, discontinuityErrorTolerance);
	float64_t2 prevPoint = float64_t2(nbl::hlsl::numeric_limits<float64_t>::infinity, nbl::hlsl::numeric_limits<float64_t>::infinity);

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
				linePoints.push_back(point);
			else if (linePoints.back() != point)
				linePoints.push_back(point);
		};

	float currentPhaseShift = lineStyle.phaseShift;
	for (uint32_t sectionIdx = 0u; sectionIdx < m_sections.size(); sectionIdx++)
	{
		const auto& section = m_sections[sectionIdx];
		
		// if gap detected
		if (sectionIdx > 0u && glm::any(glm::greaterThan(glm::abs(getSectionFirstPoint(section) - prevPoint), DiscontinuityErrorTolerance)))
			flushCurrentPolyline();

		if (section.type == ObjectType::LINE)
		{
			// calculate phase shift at each point of each line in section
			const uint32_t lineCount = section.count;
			for (uint32_t i = 0u; i < lineCount; i++)
			{
				const uint32_t currIdx = section.index + i;
				const auto& currlinePoint = m_linePoints[currIdx];
				const auto& nextLinePoint = m_linePoints[currIdx + 1u];

				if (allSolid)
				{
					pushBackToLinePoints(currlinePoint.p);
					continue;
				}

				const float64_t2 lineVector = nextLinePoint.p - currlinePoint.p;
				const float64_t lineLen = glm::length(lineVector);
				const float64_t2 lineVectorNormalized = lineVector / lineLen;

				float64_t currentTracedLen = 0.0;
				const float32_t differenceToNextDrawSectionEnd = drawSectionNormalizedLen - currentPhaseShift;
				const bool insideDrawSection = differenceToNextDrawSectionEnd > 0.0f;

				// Handle beginning of the line if it's inside a draw section and draw it partially based on currentPhaseShift
				if (insideDrawSection)
				{
					const float64_t nextDrawSectionEnd = differenceToNextDrawSectionEnd * patternLen;
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
				const float64_t lenToNextDrawBegin = differenceToNextDrawSectionBegin * patternLen;
				currentTracedLen = nbl::core::min(currentTracedLen + lenToNextDrawBegin, lineLen);
				const float64_t remainingLen = lineLen - currentTracedLen;

				// Handle the rest of the line and draw full patterns fitting inside this line, last one may partially belongs to the next line or section.
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

			if (allSolid)
				pushBackToLinePoints(m_linePoints[section.index + lineCount].p);

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

				if (allSolid)
				{
					beziers.push_back(quadBezierInfo.shape);
					continue;
				}

				// setting next phase shift based on current arc length
				nbl::hlsl::shapes::Quadratic<double> quadratic = nbl::hlsl::shapes::Quadratic<double>::constructFromBezier(quadBezierInfo.shape);
				nbl::hlsl::shapes::Quadratic<double>::ArcLengthCalculator arcLenCalc = nbl::hlsl::shapes::Quadratic<double>::ArcLengthCalculator::construct(quadratic);
				const double bezierLen = arcLenCalc.calcArcLen(1.0);

				float64_t currentTracedLen = 0.0;
				const float32_t differenceToNextDrawSectionEnd = drawSectionNormalizedLen - currentPhaseShift;
				const bool insideDrawSection = differenceToNextDrawSectionEnd > 0.0f;
				if (insideDrawSection)
				{
					const float64_t nextDrawSectionEnd = differenceToNextDrawSectionEnd * patternLen;
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
				const float64_t lenToNextDrawBegin = differenceToNextDrawSectionBegin * patternLen;
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
			
			if (allSolid)
				beziers.push_back(m_quadBeziers[section.index + quadBezierCount - 1u].shape);

			currentPolyline.addQuadBeziers({ beziers.data(), beziers.data() + beziers.size() });
			beziers.clear();
		}
		
		prevPoint = getSectionLastPoint(section);
	}

	flushCurrentPolyline();
}