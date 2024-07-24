#include "DrawResourcesFiller.h"
#include "MSDFs.h"

static constexpr uint32_t2 HatchFillPatternGlyphExtents = uint32_t2(8, 8);

void line(std::vector<CPolyline>& polylines, float64_t2 begin, float64_t2 end)
{
	std::vector<float64_t2> points = {
		begin, end
	};
	CPolyline polyline;
	polyline.addLinePoints(points);
	polylines.push_back(polyline);
}

void square(std::vector<CPolyline>& polylines, float64_t2 position, float64_t2 size = float64_t2(1, 1))
{
	std::vector<float64_t2> points = {
		float64_t2(position.x, position.y),
		float64_t2(position.x, position.y + size.y),
		float64_t2(position.x + size.x, position.y + size.y),
		float64_t2(position.x + size.x, position.y),
		float64_t2(position.x, position.y)
	};
	CPolyline polyline;
	polyline.addLinePoints(points);
	polylines.push_back(polyline);
}

void checkered(std::vector<CPolyline>& polylines)
{
	line(polylines, float64_t2(0.0, 0.0), float64_t2(4.0, 0.0));
	line(polylines, float64_t2(4.0, 0.0), float64_t2(4.0, 4.0));
	line(polylines, float64_t2(4.0, 4.0), float64_t2(0.0, 4.0));
	line(polylines, float64_t2(0.0, 4.0), float64_t2(0.0, 0.0));

	line(polylines, float64_t2(4.0, 4.0), float64_t2(8.0, 4.0));
	line(polylines, float64_t2(8.0, 4.0), float64_t2(8.0, 8.0));
	line(polylines, float64_t2(8.0, 8.0), float64_t2(4.0, 8.0));
	line(polylines, float64_t2(4.0, 8.0), float64_t2(4.0, 4.0));
}

void diamonds(std::vector<CPolyline>& polylines)
{
	{
		// Outer
		std::vector<float64_t2> points = {
			float64_t2(3.5, 8.0),
			float64_t2(7.0, 4.5),
			float64_t2(3.5, 1.0),
			float64_t2(0.0, 4.5),
			float64_t2(3.5, 8.0),
		};
		CPolyline polyline;
		polyline.addLinePoints(points);
		polylines.push_back(polyline);
	}
	{
		// Inner 
		std::vector<float64_t2> points = {
			float64_t2(3.5, 6.5),
			float64_t2(1.5, 4.5),
			float64_t2(3.5, 2.5),
			float64_t2(5.5, 4.5),
			float64_t2(3.5, 6.5)
		};

		CPolyline polyline;
		polyline.addLinePoints(points);
		polylines.push_back(polyline);
	}
}

void crossHatch(std::vector<CPolyline>& polylines)
{
	{
		// Outer
		std::vector<float64_t2> points = {
			float64_t2(3.0, 0.0),
			float64_t2(0.0, 3.0),
			float64_t2(0.0, 5.0),
			float64_t2(3.0, 8.0),
			float64_t2(5.0, 8.0),
			float64_t2(8.0, 5.0),
			float64_t2(8.0, 3.0),
			float64_t2(5.0, 0.0),
			float64_t2(3.0, 0.0),
		};
		CPolyline polyline;
		polyline.addLinePoints(points);
		polylines.push_back(polyline);
	}
	{
		// Inner 
		std::vector<float64_t2> points = {
			float64_t2(4.0, 1.0),
			float64_t2(7.0, 4.0),
			float64_t2(4.0, 7.0),
			float64_t2(1.0, 4.0),
			float64_t2(4.0, 1.0),
		};
		CPolyline polyline;
		polyline.addLinePoints(points);
		polylines.push_back(polyline);
	}
}

void hatch(std::vector<CPolyline>& polylines)
{
	CPolyline polyline;

	float64_t2 basePt0 = float64_t2(9.0, -1.0);
	float64_t2 basePt1 = float64_t2(-1.0, 9.0);
	float64_t lineDiameter = 1.5;
	float64_t lineRadius = lineDiameter / 2.0;

	{
		float64_t2 radiusOffsetTL = float64_t2(+lineRadius / 2.0, +lineRadius / 2.0);
		float64_t2 radiusOffsetBL = float64_t2(-lineRadius / 2.0, -lineRadius / 2.0);
		std::vector<float64_t2> points = {
			basePt0 + radiusOffsetTL,
			basePt0 + radiusOffsetBL, // 0
			basePt1 + radiusOffsetBL, // 1
			basePt1 + radiusOffsetTL, // 2
			basePt0 + radiusOffsetTL
		};
		polyline.addLinePoints(points);
	}
	polylines.push_back(polyline);
}

void horizontal(std::vector<CPolyline>& polylines)
{
	{
		std::vector<float64_t2> points = {
			float64_t2(0.0, 3.0),
			float64_t2(0.0, 4.0),
			float64_t2(8.0, 4.0),
			float64_t2(8.0, 3.0),
			float64_t2(0.0, 3.0),
		};
		CPolyline polyline;
		polyline.addLinePoints(points);
		polylines.push_back(polyline);
	}
	{
		std::vector<float64_t2> points = {
			float64_t2(0.0, 7.0),
			float64_t2(0.0, 8.0),
			float64_t2(8.0, 8.0),
			float64_t2(8.0, 7.0),
			float64_t2(0.0, 7.0),
		};
		CPolyline polyline;
		polyline.addLinePoints(points);
		polylines.push_back(polyline);
	}
}

void vertical(std::vector<CPolyline>& polylines)
{
	{
		std::vector<float64_t2> points = {
			float64_t2(0.0, 0.0),
			float64_t2(0.0, 8.0),
			float64_t2(1.0, 8.0),
			float64_t2(1.0, 0.0),
			float64_t2(0.0, 0.0),
		};
		CPolyline polyline;
		polyline.addLinePoints(points);
		polylines.push_back(polyline);
	}
	{
		std::vector<float64_t2> points = {
			float64_t2(4.0, 0.0),
			float64_t2(4.0, 8.0),
			float64_t2(5.0, 8.0),
			float64_t2(5.0, 0.0),
			float64_t2(4.0, 0.0),
		};
		CPolyline polyline;
		polyline.addLinePoints(points);
		polylines.push_back(polyline);
	}
}

void interwoven(std::vector<CPolyline>& polylines)
{
	{
		std::vector<float64_t2> points = {
			float64_t2(4.0, 0.0),
			float64_t2(4.0, 1.0), // 0
			float64_t2(7.0, 4.0), // 1
			float64_t2(8.0, 4.0), // 2
			float64_t2(8.0, 3.0), // 3
			float64_t2(5.0, 0.0), // 4
			float64_t2(4.0, 0.0),
		};
		CPolyline polyline;
		polyline.addLinePoints(points);
		polylines.push_back(polyline);
	}
	{
		std::vector<float64_t2> points = {
			float64_t2(3.0, 4.0),
			float64_t2(0.0, 7.0), // 0
			float64_t2(0.0, 8.0), // 1
			float64_t2(1.0, 8.0), // 2
			float64_t2(4.0, 5.0), // 3
			float64_t2(4.0, 4.0), // 4
			float64_t2(3.0, 4.0),
		};
		CPolyline polyline;
		polyline.addLinePoints(points);
		polylines.push_back(polyline);
	}
}

void reverseHatch(std::vector<CPolyline>& polylines)
{
	CPolyline polyline;

	float64_t2 basePt0 = float64_t2(-1.0, -1.0);
	float64_t2 basePt1 = float64_t2(9.0, 9.0);
	float64_t lineDiameter = 1.5;
	float64_t lineRadius = lineDiameter / 2.0;

	{
		float64_t2 radiusOffsetTL = float64_t2(-lineRadius / 2.0, +lineRadius / 2.0);
		float64_t2 radiusOffsetBL = float64_t2(+lineRadius / 2.0, -lineRadius / 2.0);
		std::vector<float64_t2> points = {
			basePt0 + radiusOffsetTL,
			basePt1 + radiusOffsetTL, // 2
			basePt1 + radiusOffsetBL, // 1
			basePt0 + radiusOffsetBL, // 0
			basePt0 + radiusOffsetTL
		};
		polyline.addLinePoints(points);
	}
	polylines.push_back(polyline);
}

void squares(std::vector<CPolyline>& polylines)
{
	CPolyline polyline;
	std::vector<float64_t2> outerSquare = {
		float64_t2(1.0, 1.0),
		float64_t2(1.0, 7.0),
		float64_t2(7.0, 7.0),
		float64_t2(7.0, 1.0),
		float64_t2(1.0, 1.0),
	};
	polyline.addLinePoints(outerSquare);
	std::vector<float64_t2> innerSquare = {
		float64_t2(2.0, 2.0),
		float64_t2(6.0, 2.0),
		float64_t2(6.0, 6.0),
		float64_t2(2.0, 6.0),
		float64_t2(2.0, 2.0),
	};
	polyline.addLinePoints(innerSquare);
	polylines.push_back(polyline);
}

void circle(std::vector<CPolyline>& polylines)
{
	CPolyline polyline;
	std::vector<float64_t2> outerSquare = {
		float64_t2(2.0, 1.0),
		float64_t2(1.0, 2.0),
		float64_t2(1.0, 6.0),
		float64_t2(2.0, 7.0),
		float64_t2(6.0, 7.0),
		float64_t2(7.0, 6.0),
		float64_t2(7.0, 2.0),
		float64_t2(6.0, 1.0),
		float64_t2(2.0, 1.0)
	};
	polyline.addLinePoints(outerSquare);
	std::vector<float64_t2> innerSquare = {
		float64_t2(2.5, 2.0),
		float64_t2(5.5, 2.0),
		float64_t2(6.0, 2.5),
		float64_t2(6.0, 5.5),
		float64_t2(5.5, 6.0),
		float64_t2(2.5, 6.0),
		float64_t2(2.0, 5.5),
		float64_t2(2.0, 2.5),
		float64_t2(2.5, 2.0),
	};
	polyline.addLinePoints(innerSquare);
	polylines.push_back(polyline);
}

void lightShaded(std::vector<CPolyline>& polylines)
{
	// Light shaded-2
	square(polylines, float64_t2(0.0, 3.0));
	square(polylines, float64_t2(0.0, 7.0));

	square(polylines, float64_t2(2.0, 1.0));
	square(polylines, float64_t2(2.0, 5.0));

	square(polylines, float64_t2(4.0, 3.0));
	square(polylines, float64_t2(4.0, 7.0));

	square(polylines, float64_t2(6.0, 1.0));
	square(polylines, float64_t2(6.0, 5.0));
}

void shaded(std::vector<CPolyline>& polylines)
{
	for (uint32_t x = 0; x < 8; x++)
	{
		for (uint32_t y = 0; y < 8; y++)
		{
			if (x % 2 != y % 2)
				square(polylines, float64_t2((double)x, (double)y));
		}
	}

}

core::smart_refctd_ptr<ICPUBuffer> generateHatchFillPatternMSDF(TextRenderer* textRenderer, HatchFillPattern fillPattern, uint32_t2 msdfExtents)
{
	std::vector<CPolyline> polylines;
	switch (fillPattern)
	{
	case HatchFillPattern::CHECKERED:
		checkered(polylines);
		break;
	case HatchFillPattern::DIAMONDS:
		diamonds(polylines);
		break;
	case HatchFillPattern::CROSS_HATCH:
		crossHatch(polylines);
		break;
	case HatchFillPattern::HATCH:
		hatch(polylines);
		break;
	case HatchFillPattern::HORIZONTAL:
		horizontal(polylines);
		break;
	case HatchFillPattern::VERTICAL:
		vertical(polylines);
		break;
	case HatchFillPattern::INTERWOVEN:
		interwoven(polylines);
		break;
	case HatchFillPattern::REVERSE_HATCH:
		reverseHatch(polylines);
		break;
	case HatchFillPattern::SQUARES:
		squares(polylines);
		break;
	case HatchFillPattern::CIRCLE:
		circle(polylines);
		break;
	case HatchFillPattern::LIGHT_SHADED:
		lightShaded(polylines);
		break;
	case HatchFillPattern::SHADED:
		shaded(polylines);
		break;
	default:
		break;
	}

	// Generate MSDFgen Shape
	msdfgen::Shape glyph;
	nbl::ext::TextRendering::GlyphShapeBuilder glyphShapeBuilder(glyph);
	for (uint32_t polylineIdx = 0; polylineIdx < polylines.size(); polylineIdx++)
	{
		auto& polyline = polylines[polylineIdx];
		for (uint32_t sectorIdx = 0; sectorIdx < polyline.getSectionsCount(); sectorIdx++)
		{
			auto& section = polyline.getSectionInfoAt(sectorIdx);
			if (section.type == ObjectType::LINE)
			{
				if (section.count == 0u) continue;

				glyphShapeBuilder.moveTo(polyline.getLinePointAt(section.index).p);
				for (uint32_t i = section.index + 1; i < section.index + section.count + 1; i++)
					glyphShapeBuilder.lineTo(polyline.getLinePointAt(i).p);
			}
		}
	}
	glyphShapeBuilder.finish();
	glyph.normalize();

	float scaleX = (1.0 / float(HatchFillPatternGlyphExtents.x)) * float(msdfExtents.x);
	float scaleY = (1.0 / float(HatchFillPatternGlyphExtents.y)) * float(msdfExtents.y);
	return textRenderer->generateShapeMSDF(glyph, MSDFPixelRange, msdfExtents, 1, float32_t2(scaleX, scaleY), float32_t2(0, 0));
}

DrawResourcesFiller::msdf_hash addMSDFFillPatternTexture(TextRenderer* textRenderer, DrawResourcesFiller& drawResourcesFiller, HatchFillPattern fillPattern, SIntendedSubmitInfo& intendedNextSubmit)
{
	const auto msdfHash = DrawResourcesFiller::hashFillPattern(fillPattern);
	auto msdfResolution = drawResourcesFiller.getMSDFResolution();
	drawResourcesFiller.addMSDFTexture(
		[textRenderer, fillPattern, msdfResolution] {
			MSDFTextureUploadInfo textureUploadInfo = {
				.cpuBuffer = std::move(generateHatchFillPatternMSDF(textRenderer, fillPattern, msdfResolution)),
				.bufferOffset = 0u,
				.imageExtent = uint32_t3(msdfResolution.x, msdfResolution.y, 1),
			};
			return textureUploadInfo;
		},
		msdfHash,
		intendedNextSubmit
	);

	return msdfHash;
}