from conans import ConanFile, CMake, tools

class CardDetectorConan(ConanFile):
    name = "CardDetector"
    version = "1.0.0"
    license = "Open Source"
    author = "Julian Massing julimassing@gmail.com"
    settings = "os", "compiler", "build_type", "arch"
    generators = "cmake"
    requires = [
    ("opencv/4.1.1@conan/stable")
    ]
    default_options = {"opencv:shared": False}
    exports_sources = "*"
    short_paths = True

    def source(self):
        pass

    def imports(self):
        self.copy("*.dll", dst="bin", src="bin", keep_path=False)

    def build(self, keep_imports=True):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
        #cmake.test()

    def package(self):
        self.copy("*.lib", dst="lib", keep_path=False)
        self.copy("*.dll", dst="bin", keep_path=False)
        self.copy("*.so", dst="lib", keep_path=False)
        self.copy("*.a", dst="lib", keep_path=False)
        self.copy("CardDetector.exe", dst="bin", src="bin", keep_path=False)

    def package_info(self):
        self.cpp_info.libs = ["opencv"]

