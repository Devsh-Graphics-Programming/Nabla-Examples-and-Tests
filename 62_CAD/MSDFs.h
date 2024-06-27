#ifndef _NBL_CAD_MSDF_H_INCLUDED_
#define _NBL_CAD_MSDF_H_INCLUDED_

#include "Polyline.h"
#include "Hatch.h"
#include "IndexAllocator.h"
#include <nbl/video/utilities/SIntendedSubmitInfo.h>
#include <nbl/core/containers/LRUCache.h>  
#include "nbl/ext/TextRendering/TextRendering.h"

enum class MSDFFillPattern: uint32_t
{
	CHECKERED,
	DIAMONDS,
	CROSS_HATCH,
	HATCH,
	HORIZONTAL,
	VERTICAL,
	INTERWOVEN,
	REVERSE_HATCH,
	SQUARES,
	CIRCLE,
	LIGHT_SHADED,
	SHADED,
	COUNT
};

enum class MSDFTextureType: uint32_t
{
	HATCH_FILL_PATTERN,
	FONT_GLYPH,
};

core::smart_refctd_ptr<ICPUBuffer> generateHatchFillPatternMSDF(TextRenderer* textRenderer, MSDFFillPattern fillPattern, uint32_t2 msdfExtents);

DrawResourcesFiller::texture_hash addMSDFFillPatternTexture(TextRenderer* textRenderer, DrawResourcesFiller& drawResourcesFiller, MSDFFillPattern fillPattern, SIntendedSubmitInfo& intendedNextSubmit);

DrawResourcesFiller::texture_hash hashFillPattern(MSDFFillPattern fillPattern);

DrawResourcesFiller::texture_hash hashFontGlyph(size_t fontHash, uint32_t glyphIndex);

#endif _NBL_CAD_MSDF_H_INCLUDED_

