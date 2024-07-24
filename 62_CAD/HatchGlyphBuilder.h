// Debug tool to Visualize Freetype Glyphs by turning them into Polyline objects

class FreetypeHatchBuilder
{
public:
	// Start a new line from here
	void moveTo(const float64_t2 to)
	{
		lastPosition = to;
	}

	// Continue the last line started with moveTo (could also use the last 
	// position from a lineTo)
	void lineTo(const float64_t2 to)
	{
		if (to != lastPosition) {
			std::vector<float64_t2> linePoints;
			linePoints.push_back(lastPosition);
			linePoints.push_back(to);
			currentPolyline.addLinePoints(linePoints);

			lastPosition = to;
		}
	}

	// Continue the last moveTo or lineTo with a quadratic bezier:
	// [last position, control, end]
	void quadratic(const float64_t2 control, const float64_t2 to)
	{
		shapes::QuadraticBezier<double> bezier = shapes::QuadraticBezier<double>::construct(
			lastPosition,
			control,
			to
		);
		currentPolyline.addQuadBeziers({ &bezier, 1 });
		lastPosition = to;
	}

	// Continue the last moveTo or lineTo with a cubic bezier:
	// [last position, control1, control2, end]
	void cubic(const float64_t2 control1, const float64_t2 control2, const float64_t2 to)
	{
		std::vector<shapes::QuadraticBezier<double>> quadBeziers;
		curves::CubicCurve myCurve(
			float64_t4(lastPosition.x, lastPosition.y, control1.x, control1.y),
			float64_t4(control2.x, control2.y, to.x, to.y)
		);

		curves::Subdivision::AddBezierFunc addToBezier = [&](shapes::QuadraticBezier<double>&& info) -> void
			{
				quadBeziers.push_back(info);
			};

		curves::Subdivision::adaptive(myCurve, 0.0, 1.0, 1e-5, addToBezier, 10u);
		currentPolyline.addQuadBeziers(quadBeziers);

		lastPosition = to;
	}

	void finish()
	{
		if (currentPolyline.getSectionsCount() > 0)
			polylines.push_back(currentPolyline);
	}

	std::vector<CPolyline> polylines;
	CPolyline currentPolyline = {};
	// Set with move to and line to
	float64_t2 lastPosition = float64_t2(0.0);
};

// TODO: Figure out what this is supposed to do
static double f26dot6ToDouble(float x)
{
	return (1 / 64. * double(x));
}

static float64_t2 ftPoint2(const FT_Vector& vector) {
	return float64_t2(f26dot6ToDouble(vector.x), f26dot6ToDouble(vector.y));
}

static int ftMoveTo(const FT_Vector* to, void* user) {
	FreetypeHatchBuilder* context = reinterpret_cast<FreetypeHatchBuilder*>(user);
	context->moveTo(ftPoint2(*to));
	return 0;
}
static int ftLineTo(const FT_Vector* to, void* user) {
	FreetypeHatchBuilder* context = reinterpret_cast<FreetypeHatchBuilder*>(user);
	context->lineTo(ftPoint2(*to));
	return 0;
}

static int ftConicTo(const FT_Vector* control, const FT_Vector* to, void* user) {
	FreetypeHatchBuilder* context = reinterpret_cast<FreetypeHatchBuilder*>(user);
	context->quadratic(ftPoint2(*control), ftPoint2(*to));
	return 0;
}

static int ftCubicTo(const FT_Vector* control1, const FT_Vector* control2, const FT_Vector* to, void* user) {
	FreetypeHatchBuilder* context = reinterpret_cast<FreetypeHatchBuilder*>(user);
	context->cubic(ftPoint2(*control1), ftPoint2(*control2), ftPoint2(*to));
	return 0;
}

static int ftMoveToMSDF(const FT_Vector* to, void* user) {
	nbl::ext::TextRendering::GlyphShapeBuilder* context = reinterpret_cast<nbl::ext::TextRendering::GlyphShapeBuilder*>(user);
	context->moveTo(ftPoint2(*to));
	return 0;
}
static int ftLineToMSDF(const FT_Vector* to, void* user) {
	nbl::ext::TextRendering::GlyphShapeBuilder* context = reinterpret_cast<nbl::ext::TextRendering::GlyphShapeBuilder*>(user);
	context->lineTo(ftPoint2(*to));
	return 0;
}

static int ftConicToMSDF(const FT_Vector* control, const FT_Vector* to, void* user) {
	nbl::ext::TextRendering::GlyphShapeBuilder* context = reinterpret_cast<nbl::ext::TextRendering::GlyphShapeBuilder*>(user);
	context->quadratic(ftPoint2(*control), ftPoint2(*to));
	return 0;
}

static int ftCubicToMSDF(const FT_Vector* control1, const FT_Vector* control2, const FT_Vector* to, void* user) {
	nbl::ext::TextRendering::GlyphShapeBuilder* context = reinterpret_cast<nbl::ext::TextRendering::GlyphShapeBuilder*>(user);
	context->cubic(ftPoint2(*control1), ftPoint2(*control2), ftPoint2(*to));
	return 0;
}

bool drawFreetypeGlyph(msdfgen::Shape& shape, FT_Library library, FT_Face face)
{
	nbl::ext::TextRendering::GlyphShapeBuilder builder(shape);
	FT_Outline_Funcs ftFunctions;
	ftFunctions.move_to = &ftMoveToMSDF;
	ftFunctions.line_to = &ftLineToMSDF;
	ftFunctions.conic_to = &ftConicToMSDF;
	ftFunctions.cubic_to = &ftCubicToMSDF;
	ftFunctions.shift = 0;
	ftFunctions.delta = 0;
	FT_Error error = FT_Outline_Decompose(&face->glyph->outline, &ftFunctions, &builder);
	if (error)
		return false;
	builder.finish();
	return true;
}
