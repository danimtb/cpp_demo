from conan import ConanFile
from conan.tools.cmake import cmake_layout, CMake
from conan.tools.build import check_min_cppstd

class ConanApplication(ConanFile):
    package_type = "application"
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    def layout(self):
        cmake_layout(self)

    def validate(self):
        check_min_cppstd(self, 20)

    def requirements(self):
        self.requires("nlohmann_json/3.11.3")
        self.requires("libtorch/2.9.1")
        self.requires("opencv/4.12.0")
        self.requires("eigen/3.4.1", override=True)
        self.requires("protobuf/6.32.1", override=True)
        self.requires("cpuinfo/cci.20251210", override=True)
        self.requires("opengl/system")
        self.requires("sdl/2.28.3")
        self.requires("sdl_image/2.8.2")
        self.requires("sdl_ttf/2.24.0")
        self.requires("libtiff/4.7.1", override=True)

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
