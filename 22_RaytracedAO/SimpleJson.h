using namespace std;

namespace simplejson
{
	enum class Type
	{
		Object,
		Array,
	};

	using State = std::pair<Type, bool>;
	using Stack = std::stack<State>;

	class Stream
	{
		Stack stack;
		stringstream buffer;
		uint32_t indent{ 0 };

	public:
		void begin_json_object();
		void end_json_object();
		void emit_json_key(const std::string& key);
		void emit_json_key_value(const std::string& key, const std::string& value);
		void emit_json_key_value(const std::string& key, bool value);
		void emit_json_key_value(const std::string& key, uint32_t value);
		void emit_json_key_value(const std::string& key, int32_t value);
		void emit_json_key_value(const std::string& key, float value);
		void emit_json_key_object(const std::string& key);
		void emit_json_key_array(const std::string& key);

		void begin_json_array();
		void end_json_array();
		void emit_json_array_value(const std::string& value);
		void emit_json_array_value(uint32_t value);
		void emit_json_array_value(bool value);

		std::string str() const
		{
			return buffer.str();
		}

	private:
		inline void statement_indent()
		{
			for (uint32_t i = 0; i < indent; i++)
				buffer << "    ";
		}

		template <typename T>
		inline void statement_inner(T&& t)
		{
			buffer << std::forward<T>(t);
		}

		template <typename T, typename... Ts>
		inline void statement_inner(T&& t, Ts &&... ts)
		{
			buffer << std::forward<T>(t);
			statement_inner(std::forward<Ts>(ts)...);
		}

		template <typename... Ts>
		inline void statement(Ts &&... ts)
		{
			statement_indent();
			statement_inner(std::forward<Ts>(ts)...);
			buffer << '\n';
		}

		template <typename... Ts>
		void statement_no_return(Ts &&... ts)
		{
			statement_indent();
			statement_inner(std::forward<Ts>(ts)...);
		}
	};
} // namespace simplejson
