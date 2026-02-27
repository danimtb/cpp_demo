from conan import ConanFile

class Recipe(ConanFile):
    settings = "os", "arch", "compiler", "build_type"
    generators = "CMakeToolchain", "CMakeConfigDeps"

    def requirements(self):
        self.requires("libtorch/2.9.1")
        self.requires("opencv/4.12.0")
        self.requires("eigen/3.4.1", override=True)
        self.requires("protobuf/6.32.1", override=True)
        self.requires("cpuinfo/cci.20251210", override=True)
