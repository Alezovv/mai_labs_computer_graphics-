# 🎓 MAI Computer Graphics Labs (Vulkan)

Учебный репозиторий по курсу компьютерной графики (МАИ). Проект построен на **C++20**, **Vulkan**, **GLFW** и **ImGUI**, с кроссплатформенной поддержкой **Linux** и **Windows**.

---

## 📁 Структура проекта

```text
├── docs/                 # ТЗ и методические материалы к лабораторным работам
│   ├── CG Lab #1.pdf
│   └── Lab #2 Camera and Lighting.pdf
├── common/               # Общий фреймворк для всех лабораторных (библиотека mai_framework)
│   ├── include/          # Заголовочные файлы (graphics_internal.hpp, application.hpp)
│   └── source/           # Инициализация Vulkan, GLFW, swapchain, depth buffer, ImGUI и main loop
├── lab1/                 # Лабораторная работа №1 (Тор, аффинные трансформации, проекции)
│   ├── CMakeLists.txt
│   ├── shaders/          # Шейдеры (shader.vert, shader.frag)
│   └── source/           # Исходный код приложения (application.cpp, torus.cpp/hpp)
├── lab2/                 # Лабораторная работа №2 (Камера и Блинн-Фонг освещение)
│   ├── shaders/
│   └── source/
├── CMakeLists.txt        # Корневой CMake (управление зависимостями и таргетами)
├── CMakePresets.json     # Пресеты конфигурации CMake
└── README.md
```

---

## 🛠️ Требования для сборки

- **Компилятор C++20**: GCC 10+ / Clang 10+ / MSVC 2019+
- **Vulkan SDK** & драйвер с поддержкой Vulkan 1.3+
- **CMake** 3.20+
- **Ninja** (или Make / Visual Studio)

Все внешние библиотеки (`glfw`, `vk-bootstrap`, `VulkanMemoryAllocator`, `imgui`, `glm`) загружаются автоматически через `FetchContent`.

---

## 🚀 Сборка и запуск

### 1. Конфигурация через пресеты

**В Linux:**
```bash
cmake --preset debug        # Сборка в Debug
# или
cmake --preset release      # Сборка в Release
```

**В Windows:**
```bash
cmake --preset mingw-debug  # Для MinGW / GCC
# или
cmake --preset msvc-debug   # Для Visual Studio
```

### 2. Сборка

Сборка всех таргетов:
```bash
cmake --build build-debug --parallel
```

Или конкретной лабораторной:
```bash
cmake --build build-debug --target lab1
# или
cmake --build build-debug --target lab2
# или
cmake --build build-debug --target lab3
```

### 3. Запуск

Шейдеры и текстуры автоматически копируются рядом с исполняемым файлом и ищутся по относительным путям, поэтому запуск работает из любой директории:
```bash
./build-debug/lab1/lab1
# или
./build-debug/lab2/lab2
# или
./build-debug/lab3/lab3
```
*(В Windows: `.\build-debug\lab1\lab1.exe` или `.\build-debug\lab1\Debug\lab1.exe`)*
