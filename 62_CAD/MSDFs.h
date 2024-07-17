#ifndef _NBL_CAD_MSDF_H_INCLUDED_
#define _NBL_CAD_MSDF_H_INCLUDED_

#include "Polyline.h"
#include "Hatch.h"
#include "IndexAllocator.h"
#include <nbl/video/utilities/SIntendedSubmitInfo.h>
#include <nbl/core/containers/LRUCache.h>  
#include "nbl/ext/TextRendering/TextRendering.h"

core::smart_refctd_ptr<ICPUBuffer> generateHatchFillPatternMSDF(TextRenderer* textRenderer, HatchFillPattern fillPattern, uint32_t2 msdfExtents);

DrawResourcesFiller::msdf_hash addMSDFFillPatternTexture(TextRenderer* textRenderer, DrawResourcesFiller& drawResourcesFiller, HatchFillPattern fillPattern, SIntendedSubmitInfo& intendedNextSubmit);

#endif _NBL_CAD_MSDF_H_INCLUDED_

