
#include "Hatch.h"
using namespace nbl::hlsl;

#include "common.hlsl"
#include <nbl/builtin/hlsl/equations/cubic.hlsl>
#include <nbl/builtin/hlsl/equations/quartic.hlsl>

namespace hatchutils {
    
	// Intended to mitigate issues with NaNs and precision by falling back to using simpler functions when the higher roots are small enough
	static std::array<double, 4> solveQuarticRoots(double a, double b, double c, double d, double e, double t_start, double t_end)
	{
		std::array<double, 4> t = { -1.0, -1.0, -1.0, -1.0 }; // only two candidates in range, ever

		const double quadCoeffMag = std::max(std::abs(d), std::abs(e));
		const double cubCoeffMag = std::max(std::abs(c), quadCoeffMag);
		if (std::abs(a) > std::max(std::abs(b), cubCoeffMag) * QUARTIC_THRESHHOLD)
		{
			auto res = equations::Quartic<double>::construct(a, b, c, d, e).computeRoots();
			memcpy(&t[0], &res.x, sizeof(double) * 4);
		}
		else if (abs(b) > quadCoeffMag)
		{
			auto res = equations::Cubic<double>::construct(b, c, d, e).computeRoots();
			memcpy(&t[0], &res.x, sizeof(double) * 3);
		}
		else
		{
			auto res = equations::Quadratic<double>::construct(c, d, e).computeRoots();
			memcpy(&t[0], &res.x, sizeof(double) * 2);
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
		// https://people.csail.mit.edu/bkph/articles/Quadratics.pdf
		double a = p0 - 2.0 * p1 + p2;
		double b = 2.0 * (p1 - p0);
		double c = p0;

		// Solve linear equation
		if (a == 0.0) return float64_t2(-c / b);

		double det = sqrt(b * b - 4 * a * c);
		if (b >= 0.0) {
			return float64_t2(
				(- b - det) / (2 * a),
				(2 * c) / (-b - det)
			);
		}
		else
		{
			return float64_t2(
				(2 * c) / (-b + det),
				(-b + det) / (2 * a)
			);
		}
		//double det = b * b - 4 * a * c;
		//double rcp = 0.5 / a;
		//
		//double detSqrt = sqrt(det) * rcp;
		//double brcp = b * rcp;
		//
		//// Solve linear equation
		//if (isinf(brcp)) return float64_t2(-c / b);
		//
		//return float64_t2(-detSqrt, detSqrt) - brcp;
	}
};

Hatch::QuadraticBezier Hatch::Segment::splitCurveRange(QuadraticBezier curve, double left, double right) const
{
    return curve.splitCurveTakeLower(right).splitCurveTakeUpper(left / right);
}

Hatch::QuadraticBezier Hatch::Segment::getSplitCurve() const
{
    return splitCurveRange(*originalBezier, t_start, t_end);
}

bool Hatch::Segment::isStraightLineConstantMajor() const
{
	auto major = (uint32_t)SelectedMajorAxis;
	return originalBezier->p[0][major] == originalBezier->p[1][major] &&
		originalBezier->p[0][major] == originalBezier->p[2][major];
}

std::array<double, 2> Hatch::Segment::intersect(const Segment& other) const
{
	auto major = (int)SelectedMajorAxis;
	auto minor = 1-major;
	// Use line intersections if one or both of the beziers are linear (a = 0)
	if (originalBezier->isLineSegment())
	{
		if (other.originalBezier->isLineSegment())
		{
			// Line/line intersection
			// TODO: make this less shit
			auto x1 = originalBezier->p[0][minor], y1 = originalBezier->p[0][major],
				x2 = originalBezier->p[2][minor], y2 = originalBezier->p[2][major],
				x3 = other.originalBezier->p[0][minor], y3 = other.originalBezier->p[0][major],
				x4 = other.originalBezier->p[2][minor], y4 = other.originalBezier->p[2][major];
			auto nx = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4), 
				ny = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4),
				d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
			auto intersectionX = nx / d, intersectionY = ny / d;

			// Return if point is on the lines
			if (d != 0.0 && 
				std::min(x1, x2) <= intersectionX && y1 <= intersectionY && std::max(x1, x2) >= intersectionX && y2 >= intersectionY &&
				std::min(x3, x4) <= intersectionX && y3 <= intersectionY && std::max(x3, x4) >= intersectionX && y4 >= intersectionY)
				return { other.originalBezier->intersectOrtho(ny / d, major), core::nan<double>() };

			return { core::nan<double>(), core::nan<double>() };
		}

		// Line/curve intersection
		// TODO: make this less shit as well
		auto tx = originalBezier->p[0][minor], ty = originalBezier->p[0][major],
			a = -atan2(originalBezier->p[2][minor] - tx, originalBezier->p[2][major] - ty);
		QuadraticBezier rotatedOther = {
			float64_t2((other.originalBezier->p[0].x - tx)* cos(a) - (other.originalBezier->p[0].y - ty) * sin(a), (other.originalBezier->p[0].x - tx) * sin(a) + (other.originalBezier->p[0].y - ty) * cos(a)),
			float64_t2((other.originalBezier->p[1].x - tx) * cos(a) - (other.originalBezier->p[1].y - ty) * sin(a), (other.originalBezier->p[1].x - tx) * sin(a) + (other.originalBezier->p[1].y - ty) * cos(a)),
			float64_t2((other.originalBezier->p[2].x - tx) * cos(a) - (other.originalBezier->p[2].y - ty) * sin(a), (other.originalBezier->p[2].x - tx) * sin(a) + (other.originalBezier->p[2].y - ty) * cos(a))
		};
		return { rotatedOther.intersectOrtho(0, major), core::nan<double>() };
	}

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

Hatch::Hatch(core::SRange<CPolyline> lines, const MajorAxis majorAxis, int32_t& debugStep, std::function<void(CPolyline, CPULineStyle)> debugOutput /* tmp */)
{
    std::stack<Segment> starts; // Next segments sorted by start points
    std::stack<double> ends; // Next end points
    double maxMajor;

    int major = (int)majorAxis;
    int minor = 1-major; // Minor = Opposite of major (X)

    auto drawDebugBezier = [&](QuadraticBezier bezier, float32_t4 color)
    {
        CPolyline outputPolyline;
        std::vector<shapes::QuadraticBezier<double>> beziers;
		shapes::QuadraticBezier<double> bezierInfo;
        bezierInfo.P0 = bezier.p[0];
        bezierInfo.P1 = bezier.p[1];
        bezierInfo.P2 = bezier.p[2];
        beziers.push_back(bezierInfo);
        outputPolyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<double>>(beziers.data(), beziers.data() + beziers.size()));

        CPULineStyle cpuLineStyle;
        cpuLineStyle.screenSpaceLineWidth = 4.0f;
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
        cpuLineStyle.screenSpaceLineWidth = 2.0f;
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
				auto addBezier = [&](QuadraticBezier bezier)
				{
					auto outputBezier = bezier;
					if (outputBezier.evaluateBezier(0.0)[major] > outputBezier.evaluateBezier(1.0)[major])
					{
						outputBezier.p[2] = bezier.p[0];
						outputBezier.p[0] = bezier.p[2];
						assert(outputBezier.evaluateBezier(0.0)[major] <= outputBezier.evaluateBezier(1.0)[major]);
					}

					//if (debugOutput)
					//	drawDebugBezier(outputBezier, float32_t4(0.0, 0.0, 0.0, 1.0));
					beziers.push_back(outputBezier);
				};

                auto section = line.getSectionInfoAt(secIdx);
                if (section.type == ObjectType::LINE)
                    {
						for (uint32_t itemIdx = section.index; itemIdx < section.index + section.count; itemIdx++)
						{
							auto begin = line.getLinePointAt(itemIdx);
							auto end = line.getLinePointAt(itemIdx + 1);
							addBezier({ 
								// TODO: it was suggested at some point the middle point should be NaN for this,
								// should we go ahead with that instead?
								{ begin.p, (begin.p + end.p) * 0.5, end.p }
							});
						}
					}
                else if (section.type == ObjectType::QUAD_BEZIER)
                {
                    for (uint32_t itemIdx = section.index; itemIdx < section.index + section.count; itemIdx ++)
                    {
                        auto lineBezier = line.getQuadBezierInfoAt(itemIdx);
                        QuadraticBezier unsplitBezier = { { lineBezier.p[0], lineBezier.p[1], lineBezier.p[2] }};
                        
                        // Beziers must be monotonically increasing along major
                        // First step: Make sure the bezier is monotonic, split it if not
                        std::array<QuadraticBezier, 2> monotonicSegments;
                        auto isMonotonic = unsplitBezier.splitIntoMajorMonotonicSegments(monotonicSegments);

                        if (isMonotonic)
                        {
                            // Already was monotonic
                            addBezier(unsplitBezier);
                            if (debugOutput)
                                drawDebugBezier(unsplitBezier, float32_t4(0.8, 0.8, 0.8, 0.2));
                        }
                        else
                        {
                            addBezier(monotonicSegments.data()[0]);
                            addBezier(monotonicSegments.data()[1]);
                            if (debugOutput)
                            {
                                drawDebugBezier(monotonicSegments.data()[0], float32_t4(0.0, 0.6, 0.0, 0.5));
                                drawDebugBezier(monotonicSegments.data()[1], float32_t4(0.0, 0.0, 0.6, 0.5));
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
        std::sort(segments.begin(), segments.end(), [&](Segment a, Segment b) { return a.originalBezier->p[0][major] > b.originalBezier->p[0][major]; });
		for (Segment& segment : segments)
		{
			starts.push(segment);
			//std::cout << "Starts: " << segment.originalBezier->p[0][major]  << "\n";
		}

        std::sort(segments.begin(), segments.end(), [&](Segment a, Segment b) { return a.originalBezier->p[2][major] > b.originalBezier->p[2][major]; });
        for (Segment& segment : segments)
		{
			ends.push(segment.originalBezier->p[2][major]);
			//std::cout << "Ends: " << segment.originalBezier->p[2][major] << "\n";
		}
        maxMajor = segments.front().originalBezier->p[2][major];
    }

    // Sweep line algorithm
	std::priority_queue<double, std::vector<double>, std::greater<double> > intersections; // Next intersection points as major coordinate
    std::vector<Segment> activeCandidates; // Set of active candidates for neighbor search in sweep line

	// if we weren't spawning quads, we could just have unsorted `vector<Bezier*>`
	auto candidateComparator = [&](const Segment& lhs, const Segment& rhs)
	{
		// btw you probably want the beziers in Quadratic At^2+B+C form, not control points
		double _lhs = lhs.originalBezier->evaluateBezier(lhs.t_start)[minor];
		double _rhs = rhs.originalBezier->evaluateBezier(rhs.t_start)[minor];

		auto len = [](float64_t2 vec)
		{
			return (double)sqrt(vec.x * vec.x + vec.y * vec.y);
		};
		double lenLhs = len(lhs.originalBezier->p[2] - lhs.originalBezier->p[0]);
		double lenRhs = len(rhs.originalBezier->p[2] - rhs.originalBezier->p[0]);
		auto minLen = std::min(lenLhs, lenRhs);

		//printf("Comparing bezier lhs = (%f, %f)..(%f, %f) rhs = (%f, %f)..(%f, %f) || minor at t_start (%f, %f) = %f < %f || scale of the curves: lhs = %f rhs = %f minLen = %f || abs(_lhs - _rhs) / minLen = %f\n",
		//	lhs.originalBezier->p[0].x, lhs.originalBezier->p[0].y,
		//	lhs.originalBezier->p[2].x, lhs.originalBezier->p[2].y,
		//	rhs.originalBezier->p[0].x, rhs.originalBezier->p[0].y,
		//	rhs.originalBezier->p[2].x, rhs.originalBezier->p[2].y,
		//	lhs.t_start, rhs.t_start,
		//	_lhs, _rhs,
		//	len(lhs.originalBezier->p[2] - lhs.originalBezier->p[0]),
		//	len(rhs.originalBezier->p[2] - rhs.originalBezier->p[0]),
		//	minLen,
		//	abs(_lhs - _rhs) / minLen
		//);
		// Threshhold here for intersection points, where the minor values for the curves are
		// very close but could be smaller, causing the curves to be in the wrong order
		// TODO: figure out if this is the best ay to do this
		if (abs(_lhs - _rhs) / minLen < 1e-3)
		{
			// this is how you want to order the derivatives dmin/dmaj=-INF dmin/dmaj = 0 dmin/dmaj=INF
			// also leverage the guarantee that `dmaj>=0` to ger numerically stable compare
			float64_t2 lTan = lhs.originalBezier->tangent(lhs.t_start);
			float64_t2 rTan = rhs.originalBezier->tangent(rhs.t_start);
			_lhs = lTan[minor] * rTan[major];
			_rhs = rTan[minor] * lTan[major];
			if (_lhs == _rhs)
			{
				// TODO: this is getting the polynominal A for the bezier
				// when bezier gets converted to A, B, C polynominal this is just ->A
				float64_t2 lAcc = lhs.originalBezier->p[0] - 2.0 * lhs.originalBezier->p[1] + lhs.originalBezier->p[2];
				float64_t2 rAcc = rhs.originalBezier->p[0] - 2.0 * rhs.originalBezier->p[1] + rhs.originalBezier->p[2];
				_lhs = lAcc[minor] * rTan[major];
				_rhs = rTan[minor] * lAcc[major];
			}
		}
		return _lhs < _rhs;
	};
	int32_t step = 0;
    auto addToCandidateSet = [&](const Segment& entry)
    {
		//std::cout << "Add to candidate set: (" << entry.originalBezier->p[0].x << ", " << entry.originalBezier->p[0].y << "),"
		//	"(" << entry.originalBezier->p[1].x << ", " << entry.originalBezier->p[1].y << ")," <<
		//	"(" << entry.originalBezier->p[2].x << ", " << entry.originalBezier->p[2].y << ")" <<
		//	"\n";
		if (entry.isStraightLineConstantMajor())
		{
			//std::cout << "Above was a straight line in major, ignored\n";
			return;
		}
        // Look for intersections among active candidates
        // this is a little O(n^2) but only in the `n=candidates.size()`
        for (const auto& segment : activeCandidates)
        {
            // find intersections entry vs segment
            auto intersectionPoints = entry.intersect(segment);
			if (step == debugStep)
			{
				//drawDebugBezier(entry.getSplitCurve(), float64_t4(1.0, 0.0, 0.0, 1.0));
				//drawDebugBezier(segment.getSplitCurve(), float64_t4(0.0, 1.0, 0.0, 1.0));

				for (uint32_t i = 0; i < intersectionPoints.size(); i++)
				{
					if (nbl::core::isnan(intersectionPoints[i]))
						continue;
					auto point = segment.originalBezier->evaluateBezier(intersectionPoints[i]);
					auto min = point - 0.3;
					auto max = point + 0.3;
					drawDebugLine(float64_t2(min.x, min.y), float64_t2(max.x, min.y), float32_t4(0.0, 0.3, 0.0, 0.8));
					drawDebugLine(float64_t2(max.x, min.y), float64_t2(max.x, max.y), float32_t4(0.0, 0.3, 0.0, 0.8));
					drawDebugLine(float64_t2(min.x, max.y), float64_t2(max.x, max.y), float32_t4(0.0, 0.3, 0.0, 0.8));
					drawDebugLine(float64_t2(min.x, min.y), float64_t2(min.x, max.y), float32_t4(0.0, 0.3, 0.0, 0.8));
				}
			}

            for (uint32_t i = 0; i < intersectionPoints.size(); i++)
            {
                if (nbl::core::isnan(intersectionPoints[i]))
                    continue;
                intersections.push(segment.originalBezier->evaluateBezier(intersectionPoints[i])[major]);

                //if (debugOutput) {
                //    auto pt = segment.originalBezier->evaluateBezier(intersectionPoints[i]);
                //    auto min = pt - float64_t2(10.0, 10.0);
                //    auto max = pt + float64_t2(10.0, 10.0);
                //    drawDebugLine(float64_t2(min.x, min.y), float64_t2(max.x, min.y), float32_t4(0.0, 0.3, 0.0, 0.1));
                //    drawDebugLine(float64_t2(max.x, min.y), float64_t2(max.x, max.y), float32_t4(0.0, 0.3, 0.0, 0.1));
                //    drawDebugLine(float64_t2(min.x, max.y), float64_t2(max.x, max.y), float32_t4(0.0, 0.3, 0.0, 0.1));
                //    drawDebugLine(float64_t2(min.x, min.y), float64_t2(min.x, max.y), float32_t4(0.0, 0.3, 0.0, 0.1));
                //}
            }
        }
		activeCandidates.push_back(entry);
    };


    double lastMajor = starts.top().originalBezier->evaluateBezier(starts.top().t_start)[major];
    //std::cout << "\n\nBegin! Max major: " << maxMajor << "\n";
    while (lastMajor!=maxMajor)
	{
		if (debugOutput && step > debugStep)
			break;
		bool isCurrentDebugStep = step == debugStep;
		
        double newMajor;
		bool addStartSegmentToCandidates = false;

        const double maxMajorEnds = ends.top();

		const Segment nextStartEvent = starts.empty() ? Segment() : starts.top();
		const double minMajorStart = nextStartEvent.originalBezier ? nextStartEvent.originalBezier->evaluateBezier(nextStartEvent.t_start)[major] : 0.0;

        // We check which event, within start, end and intersection events have the smallest
        // major coordinate at this point

		auto intersectionVisit = [&]()
		{
			auto newMajor = intersections.top();
			if (debugOutput && isCurrentDebugStep)
				drawDebugLine(float64_t2(-1000.0, newMajor), float64_t2(1000.0, newMajor), float32_t4(0.0, 0.0, 0.8, 1.0));
			intersections.pop(); // O(n)
			//std::cout << "Intersection event at " << newMajor << "\n";
			return newMajor;
		};

        // next start event is before next end event
        if (nextStartEvent.originalBezier && minMajorStart < maxMajorEnds)
        {
            // next start event is before next intersection event
            // (start event)
            if (intersections.empty() || minMajorStart < intersections.top()) // priority queue top() is O(1)
            {
                starts.pop();
                newMajor = minMajorStart;
				addStartSegmentToCandidates = true;
				if (debugOutput && isCurrentDebugStep)
                    drawDebugLine(float64_t2(-1000.0, newMajor), float64_t2(1000.0, newMajor), float32_t4(0.0, 0.8, 0.0, 1.0));
                //std::cout << "Start event at " << newMajor << "\n";
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
            if (debugOutput && isCurrentDebugStep)
                drawDebugLine(float64_t2(-1000.0, newMajor), float64_t2(1000.0, newMajor), float32_t4(0.0, 0.0, 0.8, 1.0));
            //std::cout << "End event at " << newMajor << "\n";
        }
        // spawn quads for the previous iterations if we advanced
        //std::cout << "New major: " << newMajor << " Last major: " << lastMajor << "\n";
        if (newMajor > lastMajor)
        {
			if (debugOutput && isCurrentDebugStep)
				drawDebugLine(float64_t2(-1000.0, lastMajor), float64_t2(1000.0, lastMajor), float32_t4(0.1, 0.1, 0.0, 0.5));
            // trim
            const auto candidatesSize = std::distance(activeCandidates.begin(),activeCandidates.end());
			//std::cout << "Candidates size: " << candidatesSize << "\n";
            // because n4ce works on loops, this must be true
            assert((candidatesSize % 2u)==0u);
            for (auto i=0u; i< candidatesSize;)
            {
                auto left = activeCandidates[i++];
                auto right = activeCandidates[i++];

                CurveHatchBox curveBox;

				left.t_end = left.originalBezier->intersectOrtho(newMajor, major);
				right.t_end = right.originalBezier->intersectOrtho(newMajor, major);
                auto splitCurveMin = left.getSplitCurve();
                auto splitCurveMax = right.getSplitCurve();
				assert(splitCurveMin.evaluateBezier(0.0)[major] <= splitCurveMin.evaluateBezier(1.0)[major]);
				assert(splitCurveMax.evaluateBezier(0.0)[major] <= splitCurveMax.evaluateBezier(1.0)[major]);

                auto curveMinAabb = splitCurveMin.getBezierBoundingBoxMinor();
                auto curveMaxAabb = splitCurveMax.getBezierBoundingBoxMinor();
                curveBox.aabbMin = float64_t2(std::min(curveMinAabb.first.x, curveMaxAabb.first.x), lastMajor);
                curveBox.aabbMax = float64_t2(std::max(curveMinAabb.second.x, curveMaxAabb.second.x), newMajor);

				if (isCurrentDebugStep)
				{
					drawDebugBezier(splitCurveMin, float64_t4(1.0, 0.0, 0.0, 1.0));
					drawDebugBezier(splitCurveMax, float64_t4(0.0, 1.0, 0.0, 1.0));

					//printf(std::format("Split curve min ({}, {}), ({}, {}), ({}, {}) max ({}, {}), ({}, {}), ({}, {})\n",
					//	splitCurveMin.p[0].x, splitCurveMin.p[0].y,
					//	splitCurveMin.p[1].x, splitCurveMin.p[1].y,
					//	splitCurveMin.p[2].x, splitCurveMin.p[2].y,
					//	splitCurveMax.p[0].x, splitCurveMax.p[0].y,
					//	splitCurveMax.p[1].x, splitCurveMax.p[1].y,
					//	splitCurveMax.p[2].x, splitCurveMax.p[2].y
					//).c_str());

				}

                //if (debugOutput)
                //{
                //    drawDebugLine(float64_t2(curveBox.aabbMin.x, curveBox.aabbMin.y), float64_t2(curveBox.aabbMax.x, curveBox.aabbMin.y), float32_t4(0.0, 0.3, 0.0, 0.1));
                //    drawDebugLine(float64_t2(curveBox.aabbMax.x, curveBox.aabbMin.y), float64_t2(curveBox.aabbMax.x, curveBox.aabbMax.y), float32_t4(0.0, 0.3, 0.0, 0.1));
                //    drawDebugLine(float64_t2(curveBox.aabbMin.x, curveBox.aabbMax.y), float64_t2(curveBox.aabbMax.x, curveBox.aabbMax.y), float32_t4(0.0, 0.3, 0.0, 0.1));
                //    drawDebugLine(float64_t2(curveBox.aabbMin.x, curveBox.aabbMin.y), float64_t2(curveBox.aabbMin.x, curveBox.aabbMax.y), float32_t4(0.0, 0.3, 0.0, 0.1));
                //}
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
					if (output[0].y < 1e-6) output[0].y = 0.0;
                    output[1] = 2.0 * (p1 - p0);
                    output[2] = p0;
					//printf(std::format("Transformed curve A, B, C ({}, {}), ({}, {}), ({}, {})\n",
					//	output[0].x, output[0].y,
					//	output[1].x, output[1].y,
					//	output[2].x, output[2].y
					//).c_str());
                };
                transformCurves(splitCurveMin, curveBox.aabbMin, curveBox.aabbMax, &curveBox.curveMin[0]);
                transformCurves(splitCurveMax, curveBox.aabbMin, curveBox.aabbMax, &curveBox.curveMax[0]);

				//printf(std::format("Hatch box AABB ({}, {})..({}, {}) curve min A ({}, {}) B ({}, {}) C ({}, {}) curve max A ({}, {}) B ({}, {}) C ({}, {})\n",
				//	curveBox.aabbMin.x, curveBox.aabbMin.y,
				//	curveBox.aabbMax.x, curveBox.aabbMax.y,
				//	curveBox.curveMin[0].x, curveBox.curveMin[0].y,
				//	curveBox.curveMin[1].x, curveBox.curveMin[1].y,
				//	curveBox.curveMin[2].x, curveBox.curveMin[2].y,
				//	curveBox.curveMax[0].x, curveBox.curveMax[0].y,
				//	curveBox.curveMax[1].x, curveBox.curveMax[1].y,
				//	curveBox.curveMax[2].x, curveBox.curveMax[2].y
				//).c_str());

                hatchBoxes.push_back(curveBox);
            }

            // advance and trim all of the beziers in the candidate set
            auto oit = activeCandidates.begin();
            for (auto iit = activeCandidates.begin(); iit != activeCandidates.end(); iit++)
            {
                const double evalAtMajor = iit->originalBezier->evaluateBezier(iit->t_end)[major];

				auto origBez = iit->originalBezier;
				//std::cout << "Candidate: (" << origBez->p[0].x << ", " << origBez->p[0].y << "),"
				//	"(" << origBez->p[1].x << ", " << origBez->p[1].y << ")," <<
				//	"(" << origBez->p[2].x << ", " << origBez->p[2].y << ") " <<
				//	"Evaluated at major: " << evalAtMajor;
                // if we scrolled past the end of the segment, remove it
                // (basically, we memcpy everything after something is different
                // and we skip on the memcpy for any items that are also different)
                // (this is supposedly a pattern with input/output operators)
                if (newMajor < evalAtMajor)
                {
                    const double new_t_start = iit->originalBezier->intersectOrtho(newMajor, major);

					//std::cout << " new_t_start = " << new_t_start << " minor at t_start = " << iit->originalBezier->evaluateBezier(new_t_start)[minor];
                    // little optimization (don't memcpy anything before something was removed)
                    if (oit != iit)
                        *oit = *iit;
                    oit->t_start = new_t_start;
                    oit++;
                }
				//std::cout << "\n";
            }
            // trim
            const auto newSize = std::distance(activeCandidates.begin(), oit);
            activeCandidates.resize(newSize);

			//std::cout << "New candidate size: " << newSize << "\n";
        }
		// If we had a start event, we need to add the candidate
		if (addStartSegmentToCandidates)
		{
			addToCandidateSet(nextStartEvent);
		}
		// We'll need to sort if we had a start event and added to the candidate set
		// or if we have advanced our candidate set
		if (addStartSegmentToCandidates || newMajor > lastMajor)
		{
			std::sort(activeCandidates.begin(), activeCandidates.end(), candidateComparator);
			if (newMajor > lastMajor)
				lastMajor = newMajor;
		}
		step++;
    }
	debugStep = debugStep - step;
}


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

Hatch::QuadraticBezier Hatch::QuadraticBezier::splitCurveTakeLower(double t) const
{
	QuadraticBezier outputCurve;
	outputCurve.p[0] = p[0];
	outputCurve.p[1] = (1 - t) * p[0] + t * p[1];
	outputCurve.p[2] = (1 - t) * ((1 - t) * p[0] + t * p[1]) + t * ((1 - t) * p[1] + t * p[2]);
	assert(outputCurve.evaluateBezier(0.0)[(int)SelectedMajorAxis] <= outputCurve.evaluateBezier(1.0)[(int)SelectedMajorAxis]);

	return outputCurve;
}

Hatch::QuadraticBezier Hatch::QuadraticBezier::splitCurveTakeUpper(double t) const
{
	QuadraticBezier outputCurve;
	outputCurve.p[0] = (1 - t) * ((1 - t) * p[0] + t * p[1]) + t * ((1 - t) * p[1] + t * p[2]);
	outputCurve.p[1] = (1 - t) * p[1] + t * p[2];
	outputCurve.p[2] = p[2]; 
	assert(outputCurve.evaluateBezier(0.0)[(int)SelectedMajorAxis] <= outputCurve.evaluateBezier(1.0)[(int)SelectedMajorAxis]);

	return outputCurve;
}

bool Hatch::QuadraticBezier::splitIntoMajorMonotonicSegments(std::array<Hatch::QuadraticBezier, 2>& out) const
{
	// Getting derivatives for our quadratic bezier
	auto major = (uint32_t)SelectedMajorAxis;
	auto a = 2.0 * p[1][major] - p[0][major];
	auto b = 2.0 * p[2][major] - p[1][major];

	// Finding roots for the quadratic bezier derivatives (a straight line)
	auto rcp = 1.0 / (b - a);
	auto t = -a * rcp;
	if (isinf(rcp) || t <= 0.0 || t >= 1.0) return true;
	out = { splitCurveTakeLower(t), splitCurveTakeUpper(t) };
	return false;
}

// https://pomax.github.io/bezierinfo/#boundingbox
std::pair<float64_t2, float64_t2> Hatch::QuadraticBezier::getBezierBoundingBoxMinor() const
{
	auto minor = (uint32_t)SelectedMinorAxis;
	double A = p[0][minor] - 2.0 * p[1][minor] + p[2][minor];
	double B = 2.0 * (p[1][minor] - p[0][minor]);

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

bool Hatch::QuadraticBezier::isLineSegment() const
{
	float64_t2 A = p[0] - 2.0 * p[1] + p[2];
	return abs(A.x) < 1e-6 && abs(A.y) < 1e-6;
}