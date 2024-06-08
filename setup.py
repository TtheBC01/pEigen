from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=['src/python_bindings.d/peigen.cxx'])
        self.sourcedir = os.path.abspath(sourcedir)

class BuildCMake(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        print("BUILD_TEMP: ", self.build_temp)
        print("BUILD_LIB: ", self.build_lib)
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + os.path.abspath(self.build_temp),
            '-DCMAKE_BUILD_TYPE=' + ('Debug' if self.debug else 'Release'),
            '-DCMAKE_INSTALL_PREFIX=' + os.path.abspath(self.build_lib),
        ]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        os.chdir(self.build_temp)
        self.spawn(['cmake', ext.sourcedir] + cmake_args)
        if not self.dry_run:
            self.spawn(['cmake', '--build', '.'])
        os.chdir(os.path.abspath('.'))

setup(
    name='peigen',
    version='1.0.0',
    author='Todd Chapman',
    license_files='LICENSE',
    author_email='tac688@alumni.stanford.edu',
    description='Python bindings for Eigen Tux',
    long_description='Python bindings for Eigen Tux using Boost Python.',
    packages=['peigen'],
    package_dir={"": "build/temp.linux-x86_64-cpython-312/src"},
    zip_safe=False,
    ext_modules=[CMakeExtension('peigen')],
    cmdclass=dict(build_ext=BuildCMake),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
