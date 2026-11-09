#include <cstdint>
#include <climits>
#include <iostream>
#include <filesystem>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>

#include "graphics_internal.hpp"
#include "application.hpp"

namespace {

constexpr int32_t default_window_width = 1280;
constexpr int32_t default_window_height = 720;

constexpr char default_window_title[] = "Vulkan Starter App";

GLFWwindow* glfw_window;

} // namespace

int main() {
#ifdef __linux__
	// If running inside a Snap environment (e.g. VS Code integrated terminal),
	// Snap injects conflicting GSettings and GTK environment variables that break host libraries (e.g. libdecor).
	const char* orig_data_dirs = std::getenv("XDG_DATA_DIRS_VSCODE_SNAP_ORIG");
	if (orig_data_dirs && *orig_data_dirs) {
		setenv("XDG_DATA_DIRS", orig_data_dirs, 1);
	} else if (const char* data_dirs = std::getenv("XDG_DATA_DIRS")) {
		std::string filtered;
		std::string cur(data_dirs);
		size_t pos = 0;
		while (pos < cur.size()) {
			size_t next = cur.find(':', pos);
			if (next == std::string::npos) next = cur.size();
			std::string token = cur.substr(pos, next - pos);
			if (token.find("/snap/") == std::string::npos) {
				if (!filtered.empty()) filtered += ":";
				filtered += token;
			}
			pos = next + 1;
		}
		if (!filtered.empty()) {
			setenv("XDG_DATA_DIRS", filtered.c_str(), 1);
		}
	}
	unsetenv("GSETTINGS_SCHEMA_DIR");
	unsetenv("GTK_PATH");
	unsetenv("GTK_EXE_PREFIX");
	unsetenv("GIO_MODULE_DIR");
#endif

	int status = EXIT_SUCCESS;

	if (!glfwInit()) {
		std::cerr << "Failed to initialize GLFW\n";
		return EXIT_FAILURE;
	}

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	const char* window_title = application::getTitle();
	if (!window_title || !*window_title) {
		window_title = default_window_title;
	}

	glfw_window = glfwCreateWindow(default_window_width, default_window_height,
	                               window_title, nullptr, nullptr);
	if (glfw_window == nullptr) {
		status = EXIT_FAILURE;
		goto err_null_window;
	}

	glfwSetFramebufferSizeCallback(glfw_window, [](GLFWwindow*, int width, int height){
		if (width == 0 || height == 0) {
			return;
		}

		graphics::internal::resize(width, height);
	});

	if (ImGui::CreateContext() == nullptr) {
		std::cerr << "Failed to create ImGUI context\n";
		status = EXIT_FAILURE;
		goto err_imgui_init;
	}

	{
		ImGuiIO& io = ImGui::GetIO();
		const char* font_candidates[] = {
			"/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
			"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
			"/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
			"/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
			"C:\\Windows\\Fonts\\arial.ttf",
			"C:\\Windows\\Fonts\\segoeui.ttf",
		};
		ImFont* cyrillicFont = nullptr;
		for (const char* font_path : font_candidates) {
			if (std::filesystem::exists(font_path)) {
				cyrillicFont = io.Fonts->AddFontFromFileTTF(font_path, 17.0f, nullptr, io.Fonts->GetGlyphRangesCyrillic());
				if (cyrillicFont != nullptr) {
					io.FontDefault = cyrillicFont;
					break;
				}
			}
		}
		if (!cyrillicFont) {
			io.Fonts->AddFontDefault();
		}

		ImGui::StyleColorsDark();
		ImGuiStyle& style = ImGui::GetStyle();
		style.WindowRounding = 6.0f;
		style.FrameRounding = 4.0f;
		style.PopupRounding = 4.0f;
		style.GrabRounding = 3.0f;
	}

	if (!ImGui_ImplGlfw_InitForVulkan(glfw_window, true)) {
		std::cerr << "Failed to initialize ImGUI GLFW backend for Vulkan renderer\n";
		status = EXIT_FAILURE;
		goto err_imgui_glfw_init;
	}

	if (!graphics::internal::initialize(glfw_window)) {
		std::cerr << "Failed to initialize graphics\n";
		status = EXIT_FAILURE;
		goto err_graphics_init;
	}

	if (!application::initialize()) {
		std::cerr << "Failed to initialize application\n";
		status = EXIT_FAILURE;
		goto err_application_init;
	}

	while (!glfwWindowShouldClose(glfw_window)) {
		const double time = glfwGetTime();

		glfwPollEvents();
		ImGui_ImplGlfw_NewFrame();

		ImGui::NewFrame();
		application::update(time);
		ImGui::Render();

		graphics::internal::FrameData fd = graphics::internal::prepare();
		application::render(fd);
		graphics::internal::submitAndPresent();
	}

	application::shutdown();
err_application_init:
	graphics::internal::shutdown();
err_graphics_init:
	ImGui_ImplGlfw_Shutdown();
err_imgui_glfw_init:
	ImGui::DestroyContext();
err_imgui_init:
	glfwDestroyWindow(glfw_window);
err_null_window:
	glfwTerminate();

	return status;
}