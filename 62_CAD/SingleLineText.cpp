#include "SingleLineText.h"

SingleLineText::SingleLineText(core::smart_refctd_ptr<nbl::ext::TextRendering::FontFace>&& face, const std::string& text)
{
	m_face = std::move(face);
	m_glyphBoxes.reserve(text.length());

	m_boundingBox.min = float64_t2(0.0, 0.0);
	m_boundingBox.max = float64_t2(0.0, 0.0);

	// Position transform
	float64_t2 currentPos = float32_t2(0.0, 0.0);
	for (uint32_t i = 0; i < text.length(); i++)
	{
		const auto glyphIndex = m_face->getGlyphIndex(wchar_t(text.at(i)));
		const auto glyphMetrics = m_face->getGlyphMetrics(glyphIndex);
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

			m_boundingBox.min.x = nbl::core::min(m_boundingBox.min.x, glyphBbox.topLeft.x);
			m_boundingBox.min.y = nbl::core::min(m_boundingBox.min.y, glyphBbox.topLeft.y - glyphBbox.size.y);
			m_boundingBox.max.x = nbl::core::max(m_boundingBox.max.x, glyphBbox.topLeft.x + glyphBbox.size.x);
			m_boundingBox.max.y = nbl::core::max(m_boundingBox.max.y, glyphBbox.topLeft.y);

			m_glyphBoxes.push_back(glyphBbox);
		}
		currentPos += glyphMetrics.advance;
	}
}

void SingleLineText::Draw(
	DrawResourcesFiller& drawResourcesFiller,
	SIntendedSubmitInfo& intendedNextSubmit,
	const float64_t2& baselineStart,
	const float32_t2& scale,
	const float32_t& rotateAngle,
	const float32_t4& color,
	const float32_t tiltTiltAngle,
	const float32_t boldInPixels) const
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

	// TODO: Use Separate TextStyleInfo or something, and somehow alias with line style for improved readability
	LineStyleInfo lineStyle = {};
	lineStyle.color = color;
	lineStyle.screenSpaceLineWidth = tan(tiltTiltAngle);
	lineStyle.worldSpaceLineWidth = boldInPixels;
	const uint32_t styleIdx = drawResourcesFiller.addLineStyle_SubmitIfNeeded(lineStyle, intendedNextSubmit);
	auto glyphObjectIdx = drawResourcesFiller.addMainObject_SubmitIfNeeded(styleIdx, intendedNextSubmit);

	for (const auto& glyphBox : m_glyphBoxes)
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