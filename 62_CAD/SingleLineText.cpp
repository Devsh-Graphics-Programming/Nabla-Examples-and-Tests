#include "SingleLineText.h"

SingleLineText::SingleLineText(core::smart_refctd_ptr<nbl::ext::TextRendering::FontFace>&& face, const std::string& text)
{
	m_face = std::move(face);
	glyphBoxes.reserve(text.length());

	// Position transform
	float64_t2 currentPos = float32_t2(0.0, 0.0);
	for (uint32_t i = 0; i < text.length(); i++)
	{
		const auto glyphIndex = m_face->getGlyphIndex(wchar_t(text.at(i)));
		const auto glyphMetrics = m_face->getGlyphMetricss(glyphIndex);
		const bool skipGenerateGlyph = (glyphIndex == 0 || (glyphMetrics.size.x == 0.0 && glyphMetrics.size.y == 0.0));

		if (!skipGenerateGlyph)
		{
#ifdef VERIFY_DEBUG
			msdfgen::Shape shape = m_face->generateGlyphShape(glyphIndex);
			_NBL_BREAK_IF(shape.contours.empty());
#endif
			GlyphBox glyphBbox = 
			{
				.topLeft = currentPos + glyphMetrics.horizontalBearing,
				.size = glyphMetrics.size,
				.glyphIdx = glyphIndex,
			};
			glyphBoxes.push_back(glyphBbox);
		}
		currentPos += glyphMetrics.advance;
	}
}

void SingleLineText::Draw(
	DrawResourcesFiller& drawResourcesFiller,
	SIntendedSubmitInfo& intendedNextSubmit,
	const float64_t2& baselineStart,
	const float32_t2& scale,
	const float32_t& rotateAngle)
{
	float32_t2 vec(cos(rotateAngle), sin(rotateAngle));
	float64_t3x3 rotationMulScaleMat =
	{
		vec.x  * scale.x,	vec.y * scale.y,	0.0,
		-vec.y * scale.x,	vec.x * scale.y,	0.0,
		0.0,				0.0,				1.0,
	};
	float64_t3x3 translationMat =
	{
		1.0,	0.0,	baselineStart.x,
		0.0,	1.0,	baselineStart.y,
		0.0,	0.0,	1.0,
	};
	float64_t3x3 transformation = mul(translationMat, rotationMulScaleMat);

	LineStyleInfo lineStyle = {};
	lineStyle.color = float32_t4(1.0, 1.0, 1.0, 1.0);
	const uint32_t styleIdx = drawResourcesFiller.addLineStyle_SubmitIfNeeded(lineStyle, intendedNextSubmit);
	auto glyphObjectIdx = drawResourcesFiller.addMainObject_SubmitIfNeeded(styleIdx, intendedNextSubmit);

	for (const auto& glyphBox : glyphBoxes)
	{
		const float64_t2 topLeft = mul(transformation, float64_t3(glyphBox.topLeft, 1.0)).xy;
		const float64_t2 dirU = mul(transformation, float64_t3(glyphBox.size.x, 0.0, 0.0)).xy;
		const float64_t2 dirV = mul(transformation, float64_t3(0.0, -glyphBox.size.y, 0.0)).xy;

		// float32_t3 xx = float64_t3(0.0, -glyphBox.size.y, 0.0);
		const float32_t aspectRatio = static_cast<float32_t>(glm::length(dirV) / glm::length(dirU)); // check if you can just do: (glyphBox.size.y * scale.y) / glyphBox.size.x * scale.x)
		const float32_t2 minUV = m_face->getUV(float32_t2(0.0f,0.0f), glyphBox.size, drawResourcesFiller.getMSDFResolution(), MSDFPixelRange);
		drawResourcesFiller.drawFontGlyph(m_face.get(), glyphBox.glyphIdx, topLeft, dirU, aspectRatio, minUV, glyphObjectIdx, intendedNextSubmit);
	}

}

