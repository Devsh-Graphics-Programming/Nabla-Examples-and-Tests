#include "SingleLineText.h"

SingleLineText::SingleLineText(nbl::ext::TextRendering::FontFace* face, const std::wstring& text)
{
	m_glyphBoxes.reserve(text.length());

	m_boundingBox.min = float64_t2(0.0, 0.0);
	m_boundingBox.max = float64_t2(0.0, 0.0);

	// Position transform
	float64_t2 currentPos = float32_t2(0.0, 0.0);
	for (uint32_t i = 0; i < text.length(); i++)
	{
		const auto glyphIndex = face->getGlyphIndex(text.at(i));
		const auto glyphMetrics = face->getGlyphMetrics(glyphIndex);
		const bool skipGenerateGlyph = (glyphIndex == 0 || (glyphMetrics.size.x == 0.0 && glyphMetrics.size.y == 0.0));

		if (!skipGenerateGlyph)
		{
#ifdef VERIFY_DEBUG
			msdfgen::Shape shape = face->generateGlyphShape(glyphIndex);
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
	nbl::ext::TextRendering::FontFace* face,
	const float64_t2& baselineStart,
	const float32_t2& scale,
	const float32_t& rotateAngle,
	const float32_t4& color,
	const float32_t tiltTiltAngle,
	const float32_t boldInPixels) const
{
	float32_t2 vec(cos(rotateAngle), sin(rotateAngle));
	float64_t3x3 transformation =
	{
		vec.x  * scale.x,	vec.y * scale.y,	baselineStart.x,
		-vec.y * scale.x,	vec.x * scale.y,	baselineStart.y,
		0.0,				0.0,				1.0,
	};

	// TODO: Use Separate TextStyleInfo or something, and somehow alias with line style for improved readability
	LineStyleInfo lineStyle = {};
	lineStyle.color = color;
	lineStyle.screenSpaceLineWidth = tan(tiltTiltAngle);
	lineStyle.worldSpaceLineWidth = boldInPixels;
	drawResourcesFiller.setActiveLineStyle(lineStyle);
	drawResourcesFiller.beginMainObject(MainObjectType::TEXT);

	for (const auto& glyphBox : m_glyphBoxes)
	{
		const float64_t2 topLeft = mul(transformation, float64_t3(glyphBox.topLeft, 1.0)).xy;
		const float64_t2 dirU = mul(transformation, float64_t3(glyphBox.size.x, 0.0, 0.0)).xy;
		const float64_t2 dirV = mul(transformation, float64_t3(0.0, -glyphBox.size.y, 0.0)).xy;

		// float32_t3 xx = float64_t3(0.0, -glyphBox.size.y, 0.0);
		const float32_t aspectRatio = static_cast<float32_t>(glm::length(dirV) / glm::length(dirU)); // check if you can just do: (glyphBox.size.y * scale.y) / glyphBox.size.x * scale.x)
		const float32_t2 minUV = face->getUV(float32_t2(0.0f,0.0f), glyphBox.size, drawResourcesFiller.getMSDFResolution(), MSDFPixelRange);
		drawResourcesFiller.drawFontGlyph(face, glyphBox.glyphIdx, topLeft, dirU, aspectRatio, minUV, intendedNextSubmit);
	}

	drawResourcesFiller.endMainObject();
}