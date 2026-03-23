#ifndef _PATH_TRACER_RENDER_LINEAR_ENTRYPOINTS_INCLUDED_
#define _PATH_TRACER_RENDER_LINEAR_ENTRYPOINTS_INCLUDED_

[numthreads(RenderWorkgroupSize, 1, 1)]
[shader("compute")]
void main(uint32_t3 threadID : SV_DispatchThreadID)
{
	pathtracer_render_variant::runLinear(threadID);
}

#endif
