#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>

namespace Utils {

	static std::vector<char> ReadFile(const std::string& filename)
	{
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open())
		{
			throw std::runtime_error("Failed to open file");
		}

		unsigned long long fileSize = (unsigned long long)file.tellg();
		std::vector<char> buffer(fileSize);

		file.seekg(0, std::ios::beg);
		file.read(buffer.data(), fileSize);

		if (!file)
		{
			throw std::runtime_error("Failed to read file content");
		}


		file.close();

		return buffer;
	}

} // namespace Utils