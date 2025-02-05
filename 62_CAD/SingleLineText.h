#pragma once
#include "DrawResourcesFiller.h"

using namespace nbl;
using namespace nbl::video;
using namespace nbl::core;
using namespace nbl::asset;
using namespace nbl::ext::TextRendering;

class SingleLineText
{
public:
	// constructs and fills the `glyphBoxes`
	SingleLineText(core::smart_refctd_ptr<nbl::ext::TextRendering::FontFace>&& face, const std::string& text);
	
	struct BoundingBox
	{
		float64_t2 min;
		float64_t2 max;
	};

	BoundingBox GetAABB() const { return m_boundingBox; }

	// iterates over `glyphBoxes` generates textures msdfs if failed to add to cache (through that lambda you put)
	// void Draw(DrawResourcesFiller& drawResourcesFiller, SIntendedSubmitInfo& intendedNextSubmit);
	// ! `baselineStart`, `scale` and `rotateAngle` affect the whole line 
	// ! `color`, `italicTiltAngle`, `boldInPixels` affect the rendering of each glyph
	// ! `italicTiltAngle` to emulate italic fonts
	// ! `boldInPixels` to bolden the font, don't try more than 1 pixel for various reasons and limitations with msdf ranges
	void Draw(
		DrawResourcesFiller& drawResourcesFiller,
		SIntendedSubmitInfo& intendedNextSubmit,
		const float64_t2& baselineStart = float64_t2(0.0,0.0),
		const float32_t2& scale = float64_t2(1.0f, 1.0f),
		const float32_t& rotateAngle = 0.0f,
		const float32_t4& color = float32_t4(1.0f,1.0f,1.0f,1.0f),
		const float32_t italicTilt = 0.0f,
		const float32_t boldInPixels = 0.0f) const;

protected:
	
	struct GlyphBox
	{
		float64_t2 topLeft;
		float32_t2 size;
		uint32_t glyphIdx;
		uint32_t pad;
	};
	
	BoundingBox m_boundingBox = {};
	std::vector<GlyphBox> m_glyphBoxes;
	core::smart_refctd_ptr<nbl::ext::TextRendering::FontFace> m_face;
};