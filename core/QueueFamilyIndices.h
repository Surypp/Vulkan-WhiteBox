#pragma once
#include <optional>

struct QueueFamilyIndices
{
	std::optional<unsigned int> graphicsFamily;
	std::optional<unsigned int> presentFamily; // display content on screen

	bool IsCompleted()
	{
		return graphicsFamily.has_value() && presentFamily.has_value();
	}
};