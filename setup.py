from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from math import floor
import os
import sys
import subprocess
import multiprocessing

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class BuildCMake(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)
    
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DCMAKE_INSTALL_PREFIX=' + os.path.abspath(self.build_lib),
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        # Do a parallel build
        build_args += ['--', '-j'+str(num_available_cpu_cores(2))]

        build_folder = os.path.abspath(self.build_temp)

        if not os.path.exists(build_folder):
            os.makedirs(build_folder)

        cmake_setup = ['cmake', ext.sourcedir] + cmake_args
        cmake_build = ['cmake', '--build', '.'] + build_args

        print("Building extension for Python {}".format(sys.version.split('\n',1)[0]))
        print("Invoking CMake setup: '{}'".format(' '.join(cmake_setup)))
        sys.stdout.flush()
        subprocess.check_call(cmake_setup, cwd=build_folder)
        print("Invoking CMake build: '{}'".format(' '.join(cmake_build)))
        sys.stdout.flush()
        subprocess.check_call(cmake_build, cwd=build_folder)

def num_available_cpu_cores(ram_per_build_process_in_gb):
    if 'TRAVIS' in os.environ and os.environ['TRAVIS']=='true':
        # When building on travis-ci, just use 2 cores since travis-ci limits
        # you to that regardless of what the hardware might suggest.
        return 2
    elif 'CMAKE_BUILD_PARALLEL_LEVEL' in os.environ and os.environ['CMAKE_BUILD_PARALLEL_LEVEL'].isnumeric():
        return int(os.environ['CMAKE_BUILD_PARALLEL_LEVEL'])
    try:
        mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')  
        mem_gib = mem_bytes/(1024.**3)
        num_cores = multiprocessing.cpu_count() 
        # make sure we have enough ram for each build process.
        mem_cores = int(floor(mem_gib/float(ram_per_build_process_in_gb)+0.5));
        # We are limited either by RAM or CPU cores.  So pick the limiting amount
        # and return that.
        return max(min(num_cores, mem_cores), 1)
    except ValueError:
        return 2 # just assume 2 if we can't get the os to tell us the right answer.


setup(
    name='peigen',
    version='1.0.0',
    author='Todd Chapman',
    license_files='LICENSE',
    author_email='tac688@alumni.stanford.edu',
    description='Python bindings for Eigen Tux',
    long_description='Python bindings for Eigen Tux using Boost Python.',
    include_package_data=True,
    zip_safe=False,
    ext_modules=[CMakeExtension('peigen','.')],
    cmdclass=dict(build_ext=BuildCMake),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    packages=find_packages(exclude=['test','eigen']),
    python_requires='>=3.6',
)
