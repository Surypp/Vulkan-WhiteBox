#pragma once
#include <vulkan/vulkan.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>
#include <png.h>
#include "GpuImage.h"
#include "GpuBuffer.h"
#include "UploadContext.h"

// --- Texture ---
// owns GpuImage + VkSampler. PNG loading via libpng.

class Texture {
public:
    Texture() = default;
    ~Texture() { Destroy(); }

    Texture(const Texture&)            = delete;
    Texture& operator=(const Texture&) = delete;

    Texture(Texture&& o) noexcept
        : _image(std::move(o._image)), _sampler(o._sampler), _device(o._device)
    { o._sampler = VK_NULL_HANDLE; }

    Texture& operator=(Texture&& o) noexcept {
        if (this != &o) { Destroy(); _image = std::move(o._image);
            _sampler = o._sampler; _device = o._device;
            o._sampler = VK_NULL_HANDLE; }
        return *this;
    }

    VkImageView View()    const { return _image.View(); }
    VkSampler   Sampler() const { return _sampler; }
    bool        IsValid() const { return _image.IsValid(); }

    static Texture LoadFromFile(
        VkDevice device, VkPhysicalDevice physDev,
        const std::string& path,
        VkCommandPool commandPool, VkQueue graphicsQueue)
    {
        uint32_t width, height;
        std::vector<unsigned char> pixels = LoadPng(path, width, height);

        VkDeviceSize imageSize = (VkDeviceSize)width * height * 4; // RGBA

        GpuBuffer staging(device, physDev, imageSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        staging.Map();
        staging.Write(pixels.data(), imageSize);
        staging.Unmap();

        GpuImage image = GpuImage::Create2D(device, physDev,
            width, height,
            VK_FORMAT_R8G8B8A8_SRGB,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT);

        UploadContext ctx(device, commandPool, graphicsQueue);

        image.TransitionLayout(ctx,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        image.CopyFromBuffer(ctx, staging.Handle(), width, height);

        image.TransitionLayout(ctx,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        VkSampler sampler = CreateSampler(device, physDev);

        Texture tex;
        tex._device  = device;
        tex._image   = std::move(image);
        tex._sampler = sampler;
        return tex;
    }

private:
    GpuImage  _image;
    VkSampler _sampler = VK_NULL_HANDLE;
    VkDevice  _device  = VK_NULL_HANDLE;

    void Destroy() {
        if (_sampler != VK_NULL_HANDLE) {
            vkDestroySampler(_device, _sampler, nullptr);
            _sampler = VK_NULL_HANDLE;
        }
    }

    static std::vector<unsigned char> LoadPng(const std::string& path,
                                              uint32_t& outWidth,
                                              uint32_t& outHeight)
    {
        FILE* fp = fopen(path.c_str(), "rb");
        if (!fp) throw std::runtime_error("Texture: cannot open " + path);

        unsigned char sig[8];
        fread(sig, 1, 8, fp);
        if (!png_check_sig(sig, 8)) {
            fclose(fp);
            throw std::runtime_error("Texture: not a PNG file: " + path);
        }

        png_structp png  = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
        png_infop   info = png_create_info_struct(png);
        if (!png || !info) { fclose(fp); throw std::runtime_error("Texture: libpng init failed"); }

        if (setjmp(png_jmpbuf(png))) {
            png_destroy_read_struct(&png, &info, nullptr);
            fclose(fp);
            throw std::runtime_error("Texture: error reading " + path);
        }

        png_init_io(png, fp);
        png_set_sig_bytes(png, 8);
        png_read_info(png, info);

        outWidth  = png_get_image_width(png, info);
        outHeight = png_get_image_height(png, info);

        // normalize to RGBA 8bpp regardless of source format
        if (png_get_bit_depth(png, info) == 16) png_set_strip_16(png);
        png_byte ct = png_get_color_type(png, info);
        if (ct == PNG_COLOR_TYPE_PALETTE)                     png_set_palette_to_rgb(png);
        if (ct == PNG_COLOR_TYPE_GRAY && png_get_bit_depth(png,info) < 8) png_set_expand_gray_1_2_4_to_8(png);
        if (png_get_valid(png, info, PNG_INFO_tRNS))          png_set_tRNS_to_alpha(png);
        if (ct == PNG_COLOR_TYPE_RGB  || ct == PNG_COLOR_TYPE_GRAY ||
            ct == PNG_COLOR_TYPE_PALETTE)                     png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
        if (ct == PNG_COLOR_TYPE_GRAY || ct == PNG_COLOR_TYPE_GRAY_ALPHA) png_set_gray_to_rgb(png);
        png_read_update_info(png, info);

        std::vector<png_bytep> rows(outHeight);
        std::vector<std::vector<png_byte>> rowData(outHeight);
        for (uint32_t y = 0; y < outHeight; y++) {
            rowData[y].resize(png_get_rowbytes(png, info));
            rows[y] = rowData[y].data();
        }
        png_read_image(png, rows.data());
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(fp);

        std::vector<unsigned char> pixels(outWidth * outHeight * 4);
        for (uint32_t y = 0; y < outHeight; y++)
            memcpy(pixels.data() + y * outWidth * 4, rows[y], outWidth * 4);
        return pixels;
    }

    // NEAREST + CLAMP_TO_EDGE for pixel art; physDev reserved for future anisotropy
    static VkSampler CreateSampler(VkDevice device, VkPhysicalDevice physDev) {
        (void)physDev;

        VkSamplerCreateInfo info{};
        info.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        info.magFilter    = VK_FILTER_NEAREST;
        info.minFilter    = VK_FILTER_NEAREST;
        info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.anisotropyEnable        = VK_FALSE;
        info.maxAnisotropy           = 1.0f;
        info.borderColor             = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        info.unnormalizedCoordinates = VK_FALSE;
        info.compareEnable           = VK_FALSE;
        info.compareOp               = VK_COMPARE_OP_ALWAYS;
        info.mipmapMode              = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        info.mipLodBias              = 0.0f;
        info.minLod                  = 0.0f;
        info.maxLod                  = 0.0f;

        VkSampler sampler;
        if (vkCreateSampler(device, &info, nullptr, &sampler) != VK_SUCCESS)
            throw std::runtime_error("Texture: vkCreateSampler failed");
        return sampler;
    }
};
