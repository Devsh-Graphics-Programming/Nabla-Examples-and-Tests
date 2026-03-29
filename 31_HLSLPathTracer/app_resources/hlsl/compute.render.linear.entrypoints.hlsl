#ifndef PATH_TRACER_ENTRYPOINT_NAME
#define PATH_TRACER_ENTRYPOINT_NAME main
#endif

#ifndef PATH_TRACER_ENTRYPOINT_POLYGON_METHOD
#define PATH_TRACER_ENTRYPOINT_POLYGON_METHOD pathtracer_variant_config::EntryPointPolygonMethod
#endif

#if !PATH_TRACER_ENABLE_LINEAR
#error Linear entrypoint requested while PATH_TRACER_ENABLE_LINEAR is disabled
#endif

[numthreads(RenderWorkgroupSize, 1, 1)]
[shader("compute")]
void PATH_TRACER_ENTRYPOINT_NAME(uint32_t3 threadID : SV_DispatchThreadID)
{
	pathtracer_render_variant::runLinear<PATH_TRACER_ENTRYPOINT_POLYGON_METHOD>(threadID);
}

#undef PATH_TRACER_ENTRYPOINT_POLYGON_METHOD
#undef PATH_TRACER_ENTRYPOINT_NAME
