#ifndef SERIALIZATION_H
#define SERIALIZATION_H

// STL 标准库
#include <string>
#include <fstream>
// 3rd party 第三方库
#include <cereal/archives/Binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>


/* 使用说明
	int main() {
		using value_type = std::vector<int>;

		// _cereal::dump< value_type >("image.Binary", data);
		
		auto data = _cereal::load< value_type >("image.Binary");

		return 0;
	}
*/

namespace _cereal {
	template<typename T>
	void dump_pair(const std::string& file_name, const T& data) {
		std::ofstream file(file_name.c_str());
		cereal::BinaryOutputArchive archive(file);

		archive(CEREAL_NVP(data.first), CEREAL_NVP(data.second));
	}

	template<typename T>
	void dump(const std::string& file_name, const T& data) {
		std::ofstream file(file_name.c_str());
		cereal::BinaryOutputArchive archive(file);

		archive(CEREAL_NVP(data));
	}

	template<typename T>
	T load_pair(const std::string& file_name) {
		T data;
		std::ifstream file(file_name.c_str());
		cereal::BinaryInputArchive archive(file);
		archive(CEREAL_NVP(data.first), CEREAL_NVP(data.second));
		return data;
	}

	template<typename T>
	T load(const std::string& file_name) {
		T data;
		std::ifstream file(file_name.c_str());
		cereal::BinaryInputArchive archive(file);
		archive(CEREAL_NVP(data));
		return data;
	}
}

#endif // SERIALIZATION_H

