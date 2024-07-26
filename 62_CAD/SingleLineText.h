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

	// iterates over `glyphBoxes` generates textures msdfs if failed to add to cache (through that lambda you put)
	// void Draw(DrawResourcesFiller& drawResourcesFiller, SIntendedSubmitInfo& intendedNextSubmit);
	void Draw(
		DrawResourcesFiller& drawResourcesFiller,
		SIntendedSubmitInfo& intendedNextSubmit,
		const float64_t2& baselineStart = float64_t2(0.0,0.0),
		const float32_t2& scale = float64_t2(1.0, 1.0),
		const float32_t& rotateAngle = 0);

protected:
	
	struct GlyphBox
	{
		float64_t2 topLeft;
		float32_t2 size;
		uint32_t glyphIdx;
		uint32_t pad;
	};

	std::vector<GlyphBox> glyphBoxes;
	core::smart_refctd_ptr<nbl::ext::TextRendering::FontFace> m_face;
};

