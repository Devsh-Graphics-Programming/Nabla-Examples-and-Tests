#include "SimpleJson.h"

using namespace simplejson;

// Hackery to emit JSON without using nlohmann/json C++ library (which requires a
// higher level of compiler compliance than is required by SPIRV-Cross
void Stream::begin_json_array()
{
	if (!stack.empty() && stack.top().second)
	{
		statement_inner(",\n");
	}
	statement("[");
	++indent;
	stack.emplace(Type::Array, false);
}

void Stream::end_json_array()
{
	if (stack.empty() || stack.top().first != Type::Array)
		std::cerr << "Invalid JSON state";
	if (stack.top().second)
	{
		statement_inner("\n");
	}
	--indent;
	statement_no_return("]");
	stack.pop();
	if (!stack.empty())
	{
		stack.top().second = true;
	}
}

void Stream::emit_json_array_value(const std::string& value)
{
	if (stack.empty() || stack.top().first != Type::Array)
		std::cerr << "Invalid JSON state";

	if (stack.top().second)
		statement_inner(",\n");

	statement_no_return("\"", value, "\"");
	stack.top().second = true;
}

void Stream::emit_json_array_value(uint32_t value)
{
	if (stack.empty() || stack.top().first != Type::Array)
		std::cerr << "Invalid JSON state";
	if (stack.top().second)
		statement_inner(",\n");
	statement_no_return(std::to_string(value));
	stack.top().second = true;
}

void Stream::emit_json_array_value(bool value)
{
	if (stack.empty() || stack.top().first != Type::Array)
		std::cerr << "Invalid JSON state";
	if (stack.top().second)
		statement_inner(",\n");
	statement_no_return(value ? "true" : "false");
	stack.top().second = true;
}

void Stream::begin_json_object()
{
	if (!stack.empty() && stack.top().second)
	{
		statement_inner(",\n");
	}
	statement("{");
	++indent;
	stack.emplace(Type::Object, false);
}

void Stream::end_json_object()
{
	if (stack.empty() || stack.top().first != Type::Object)
		std::cerr << "Invalid JSON state";
	if (stack.top().second)
	{
		statement_inner("\n");
	}
	--indent;
	statement_no_return("}");
	stack.pop();
	if (!stack.empty())
	{
		stack.top().second = true;
	}
}

void Stream::emit_json_key(const std::string& key)
{
	if (stack.empty() || stack.top().first != Type::Object)
		std::cerr << "Invalid JSON state";

	if (stack.top().second)
		statement_inner(",\n");
	statement_no_return("\"", key, "\" : ");
	stack.top().second = true;
}

void Stream::emit_json_key_value(const std::string& key, const std::string& value)
{
	emit_json_key(key);
	statement_inner("\"", value, "\"");
}

void Stream::emit_json_key_value(const std::string& key, uint32_t value)
{
	emit_json_key(key);
	statement_inner(value);
}

void Stream::emit_json_key_value(const std::string& key, int32_t value)
{
	emit_json_key(key);
	statement_inner(value);
}

void Stream::emit_json_key_value(const std::string& key, float value)
{
	emit_json_key(key);
	statement_inner(to_string(value));
}

void Stream::emit_json_key_value(const std::string& key, bool value)
{
	emit_json_key(key);
	statement_inner(value ? "true" : "false");
}

void Stream::emit_json_key_object(const std::string& key)
{
	emit_json_key(key);
	statement_inner("{\n");
	++indent;
	stack.emplace(Type::Object, false);
}

void Stream::emit_json_key_array(const std::string& key)
{
	emit_json_key(key);
	statement_inner("[\n");
	++indent;
	stack.emplace(Type::Array, false);
}