#include "Polyline.h"
#include "Hatch.h"
#include "IndexAllocator.h"
#include <nbl/video/utilities/SIntendedSubmitInfo.h>
#include <nbl/core/containers/LRUCache.h>  

enum class MsdfFillPattern: uint32_t
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

enum class MsdfTextureType: uint32_t
{
	HATCH_FILL_PATTERN,
	FONT_GLYPH,
};

struct MsdfTextureHash 
{
	MsdfTextureType textureType;
	union {
		MsdfFillPattern fillPattern;
		std::size_t glyphHash;
	};
};

template<>
struct std::hash<MsdfTextureHash>
{
    std::size_t operator()(const MsdfTextureHash& s) const noexcept
    {
		std::size_t finalHash = std::hash<uint32_t>{}(uint32_t(s.textureType));
		std::size_t textureHash;

		switch (s.textureType) 
		{
		case MsdfTextureType::HATCH_FILL_PATTERN:
			textureHash = std::hash<uint32_t>{}(uint32_t(s.fillPattern));
			break;
		case MsdfTextureType::FONT_GLYPH:
			textureHash = s.glyphHash;
			break;
		}

		nbl::core::hash_combine(finalHash, textureHash);
		return finalHash;
    }
};
 
DrawResourcesFiller::MsdfTextureUploadInfo generateHatchFillPatternMsdf(MsdfFillPattern fillPattern, uint32_t2 msdfExtents);

DrawResourcesFiller::texture_hash addMsdfFillPatternTexture(DrawResourcesFiller& drawResourcesFiller, MsdfFillPattern fillPattern, SIntendedSubmitInfo& intendedNextSubmit);

