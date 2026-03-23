from conan import ConanFile
from conan.tools.cmake import cmake_layout, CMake

class Recipe(ConanFile):
    settings = "os", "arch", "compiler", "build_type"
    generators = "CMakeToolchain", "CMakeConfigDeps"

    def layout(self):
        cmake_layout(self)

    def requirements(self):
        self.requires("libtorch/2.9.1")
        self.requires("opencv/4.12.0")
        self.requires("cuda-toolkit/12.6.0")
        
        # PicRoutette-specific dependencies
        self.requires("opengl/system")
        self.requires("sdl/2.28.3")
        self.requires("sdl_image/2.8.2")
        self.requires("sdl_ttf/2.24.0")
        self.requires("nlohmann_json/3.11.3")

        self.requires("libtiff/4.7.1", override=True)


    def build_requirements(self):
        self.tool_requires("cuda-toolkit/<host_version>")

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
        if not self.conf.get("tools.build:skip_test", True):
            cmake.ctest()

