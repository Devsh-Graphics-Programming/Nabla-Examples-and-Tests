#ifndef PATH_TRACER_ENTRYPOINT_NAME
#define PATH_TRACER_ENTRYPOINT_NAME mainPersistent
#endif

#ifndef PATH_TRACER_ENTRYPOINT_POLYGON_METHOD
#define PATH_TRACER_ENTRYPOINT_POLYGON_METHOD PPM_APPROX_PROJECTED_SOLID_ANGLE
#endif

#if !PATH_TRACER_ENABLE_PERSISTENT
#error Persistent entrypoint requested while PATH_TRACER_ENABLE_PERSISTENT is disabled
#endif

[numthreads(RenderWorkgroupSize, 1, 1)]
[shader("compute")]
void PATH_TRACER_ENTRYPOINT_NAME(uint32_t3 threadID : SV_DispatchThreadID)
{
	pathtracer_render_variant::runPersistent<PATH_TRACER_ENTRYPOINT_POLYGON_METHOD>();
}

#undef PATH_TRACER_ENTRYPOINT_POLYGON_METHOD
#undef PATH_TRACER_ENTRYPOINT_NAME
