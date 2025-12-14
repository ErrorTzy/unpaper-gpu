# SPDX-FileCopyrightText: 2020 The unpaper authors
#
# SPDX-License-Identifier: MIT

project('unpaper', 'c', version : '7.0.0', default_options : ['c_std=c11'])

if not meson.version().version_compare('>0.57')
    error('Meson 0.57 or later required')
endif

add_global_arguments('-D_POSIX_C_SOURCE=200809L', language : 'c')

cc = meson.get_compiler('c')

if not cc.has_function('strdup', prefix : '#include <string.h>', args : ['-D_POSIX_C_SOURCE=200809L'])
    error('A C library compatible with POSIX.1-2008 is required.')
endif

if cc.has_argument('-Werror=int-conversion')
    add_project_arguments('-Werror=int-conversion', language : 'c')
endif

unpaper_deps = [
    dependency('libavformat'), dependency('libavcodec'), dependency('libavutil'),
    cc.find_library('m', required : false)
]

conf_data = configuration_data()
conf_data.set('version', meson.project_version())
configure_file(input: 'version.h.in', output: 'version.h', configuration: conf_data)

cuda_opt = get_option('cuda')
have_cuda = false
nvcc = disabler()
unpaper_c_args = []
unpaper_cpp_args = []
unpaper_sources = [
    'file.c', 'parse.c', 'unpaper.c',
    'imageprocess/blit.c',
    'imageprocess/backend.c',
    'imageprocess/deskew.c',
    'imageprocess/interpolate.c',
    'imageprocess/fill.c',
    'imageprocess/filters.c',
    'imageprocess/image.c',
    'imageprocess/image_cuda.c',
    'imageprocess/masks.c',
    'imageprocess/pixel.c',
    'imageprocess/primitives.c',
    'lib/logging.c',
    'lib/perf.c',
    'lib/options.c',
    'lib/physical.c',
]

# OpenCV and CUDA setup - OpenCV is mandatory when CUDA is enabled
have_opencv = false
opencv_dep = disabler()
cpp_enabled = false
cuda_include_dir = ''

if cuda_opt.enabled()
    nvcc_compat = find_program('/usr/local/cuda-13.0/bin/nvcc', required: false)
    nvcc = nvcc_compat.found() ? nvcc_compat : find_program('nvcc', required: true)
    have_cuda = true
    if nvcc_compat.found()
        cuda_include_dir = '/usr/local/cuda-13.0/include'
    else
        cuda_include_dir = '/usr/local/cuda/include'
    endif
elif cuda_opt.auto()
    nvcc_compat = find_program('/usr/local/cuda-13.0/bin/nvcc', required: false)
    nvcc = nvcc_compat.found() ? nvcc_compat : find_program('nvcc', required: false)
    if nvcc.found()
        have_cuda = true
        if nvcc_compat.found()
            cuda_include_dir = '/usr/local/cuda-13.0/include'
        else
            cuda_include_dir = '/usr/local/cuda/include'
        endif
    endif
endif

# When CUDA is enabled, set up OpenCV (mandatory) and CUDA sources
if have_cuda
    # OpenCV is mandatory for CUDA builds
    cpp_enabled = add_languages('cpp', required : true)
    opencv_dep = dependency('opencv4', method : 'pkg-config', required : true)
    have_opencv = true

    unpaper_c_args += ['-DUNPAPER_WITH_CUDA=1', '-DUNPAPER_WITH_OPENCV=1']
    unpaper_cpp_args += ['-DUNPAPER_WITH_OPENCV=1']
    unpaper_c_args += ['-I' + cuda_include_dir]
    unpaper_cpp_args += ['-I' + cuda_include_dir]

    unpaper_sources += [
        'imageprocess/backend_cuda.c',
        'imageprocess/cuda_runtime.c',
        'imageprocess/opencv_bridge.cpp',
        'imageprocess/opencv_ops.cpp',
    ]
    unpaper_deps += [cc.find_library('dl', required : false)]

    # Link cudart for CUDA Runtime API (required for OpenCV compatibility)
    cudart_dep = cc.find_library('cudart', dirs : [
        '/usr/local/cuda-13.0/lib64',
        '/usr/local/cuda/lib64',
        '/usr/lib/x86_64-linux-gnu'
    ], required : true)
    unpaper_deps += [cudart_dep]
endif

if have_cuda
    python3 = find_program('python3', required: true)

    cuda_kernels_ptx = custom_target(
        'cuda_kernels_ptx',
        input: 'imageprocess/cuda_kernels.cu',
        output: 'cuda_kernels.ptx',
        command: [
            nvcc,
            '--ptx',
            '--fmad=false',
            '-o', '@OUTPUT@',
            '@INPUT@',
            '-arch=compute_75',
            '-I', meson.project_source_root(),
            '-Xcompiler=-U_GNU_SOURCE',
        ],
    )

    cuda_kernels_ptx_c = custom_target(
        'cuda_kernels_ptx_c',
        input: cuda_kernels_ptx,
        output: 'cuda_kernels_ptx.c',
        command: [python3, meson.project_source_root() + '/build-scripts/ptx2c.py', '@INPUT@', '@OUTPUT@'],
    )

    unpaper_sources += [cuda_kernels_ptx_c]
endif

unpaper_all_deps = have_opencv ? unpaper_deps + [opencv_dep] : unpaper_deps

unpaper = executable(
    'unpaper',
    unpaper_sources,
    c_args : unpaper_c_args,
    cpp_args : cpp_enabled ? unpaper_cpp_args : [],
    dependencies : unpaper_all_deps,
    install : true,
)

sphinx = find_program('sphinx-build', required: true, version: '>= 3.4')

custom_target(
    'man',
    command: [sphinx, '-b', 'man', join_paths(meson.source_root(), 'doc'), '@OUTDIR@'],
    input: ['doc/conf.py', 'doc/index.rst', 'doc/unpaper.1.rst'],
    output: 'unpaper.1',
    install: true,
    install_dir: join_paths(get_option('prefix'), get_option('mandir'), 'man1'),
)

pymod = import('python')
python = pymod.find_installation(required: false, modules: ['pytest', 'PIL'], disabler: true)

if not python.found()
    warning('pytest or Pillow not found, tests will not be performed.')
endif

test(
    'pytest suite',
    python,
    args: [
        '-m', 'pytest',
        '--basetemp=' + meson.current_build_dir() + '/test_output/',
        meson.project_source_root() + '/tests/unpaper_tests.py'
    ],
    env: [
        'TEST_IMGSRC_DIR=' + meson.project_source_root() + '/tests/source_images/',
        'TEST_GOLDEN_DIR=' + meson.project_source_root() + '/tests/golden_images/',
        'TEST_UNPAPER_BINARY=' + unpaper.full_path(),
    ],
    timeout : -1,
)

if have_cuda
    cuda_roundtrip_test = executable(
        'cuda_roundtrip_test',
        [
            'tests/cuda_roundtrip_test.c',
            'imageprocess/image_cuda.c',
            'imageprocess/cuda_runtime.c',
            'lib/logging.c',
        ],
        c_args : unpaper_c_args,
        dependencies : unpaper_deps,
        install : false,
    )

    test('cuda roundtrip', cuda_roundtrip_test)

    cuda_primitives_test = executable(
        'cuda_primitives_test',
        [
            'tests/cuda_primitives_test.c',
            'imageprocess/backend.c',
            'imageprocess/backend_cuda.c',
            'imageprocess/blit.c',
            'imageprocess/deskew.c',
            'imageprocess/interpolate.c',
            'imageprocess/fill.c',
            'imageprocess/filters.c',
            'imageprocess/image.c',
            'imageprocess/image_cuda.c',
            'imageprocess/masks.c',
            'imageprocess/pixel.c',
            'imageprocess/primitives.c',
            'imageprocess/cuda_runtime.c',
            'imageprocess/opencv_bridge.cpp',
            'imageprocess/opencv_ops.cpp',
            cuda_kernels_ptx_c,
            'lib/logging.c',
        ],
        c_args : unpaper_c_args,
        cpp_args : cpp_enabled ? unpaper_cpp_args : [],
        dependencies : unpaper_all_deps,
        install : false,
    )

    test('cuda primitives', cuda_primitives_test)

    cuda_resize_test = executable(
        'cuda_resize_test',
        [
            'tests/cuda_resize_test.c',
            'imageprocess/backend.c',
            'imageprocess/backend_cuda.c',
            'imageprocess/blit.c',
            'imageprocess/deskew.c',
            'imageprocess/fill.c',
            'imageprocess/filters.c',
            'imageprocess/interpolate.c',
            'imageprocess/image.c',
            'imageprocess/image_cuda.c',
            'imageprocess/masks.c',
            'imageprocess/pixel.c',
            'imageprocess/primitives.c',
            'imageprocess/cuda_runtime.c',
            'imageprocess/opencv_bridge.cpp',
            'imageprocess/opencv_ops.cpp',
            cuda_kernels_ptx_c,
            'lib/logging.c',
        ],
        c_args : unpaper_c_args,
        cpp_args : cpp_enabled ? unpaper_cpp_args : [],
        dependencies : unpaper_all_deps,
        install : false,
    )

    test('cuda resize', cuda_resize_test)

    cuda_filters_test = executable(
        'cuda_filters_test',
        [
            'tests/cuda_filters_test.c',
            'imageprocess/backend.c',
            'imageprocess/backend_cuda.c',
            'imageprocess/blit.c',
            'imageprocess/deskew.c',
            'imageprocess/fill.c',
            'imageprocess/filters.c',
            'imageprocess/interpolate.c',
            'imageprocess/image.c',
            'imageprocess/image_cuda.c',
            'imageprocess/masks.c',
            'imageprocess/pixel.c',
            'imageprocess/primitives.c',
            'imageprocess/cuda_runtime.c',
            'imageprocess/opencv_bridge.cpp',
            'imageprocess/opencv_ops.cpp',
            cuda_kernels_ptx_c,
            'lib/logging.c',
        ],
        c_args : unpaper_c_args,
        cpp_args : cpp_enabled ? unpaper_cpp_args : [],
        dependencies : unpaper_all_deps,
        install : false,
    )

    test('cuda filters', cuda_filters_test)

    cuda_masks_border_test = executable(
        'cuda_masks_border_test',
        [
            'tests/cuda_masks_border_test.c',
            'imageprocess/backend.c',
            'imageprocess/backend_cuda.c',
            'imageprocess/blit.c',
            'imageprocess/deskew.c',
            'imageprocess/fill.c',
            'imageprocess/filters.c',
            'imageprocess/interpolate.c',
            'imageprocess/image.c',
            'imageprocess/image_cuda.c',
            'imageprocess/masks.c',
            'imageprocess/pixel.c',
            'imageprocess/primitives.c',
            'imageprocess/cuda_runtime.c',
            'imageprocess/opencv_bridge.cpp',
            'imageprocess/opencv_ops.cpp',
            cuda_kernels_ptx_c,
            'lib/logging.c',
        ],
        c_args : unpaper_c_args,
        cpp_args : cpp_enabled ? unpaper_cpp_args : [],
        dependencies : unpaper_all_deps,
        install : false,
    )

    test('cuda masks/borders', cuda_masks_border_test)

    cuda_deskew_test = executable(
        'cuda_deskew_test',
        [
            'tests/cuda_deskew_test.c',
            'imageprocess/backend.c',
            'imageprocess/backend_cuda.c',
            'imageprocess/blit.c',
            'imageprocess/deskew.c',
            'imageprocess/fill.c',
            'imageprocess/filters.c',
            'imageprocess/interpolate.c',
            'imageprocess/image.c',
            'imageprocess/image_cuda.c',
            'imageprocess/masks.c',
            'imageprocess/pixel.c',
            'imageprocess/primitives.c',
            'imageprocess/cuda_runtime.c',
            'imageprocess/opencv_bridge.cpp',
            'imageprocess/opencv_ops.cpp',
            cuda_kernels_ptx_c,
            'lib/logging.c',
        ],
        c_args : unpaper_c_args,
        cpp_args : cpp_enabled ? unpaper_cpp_args : [],
        dependencies : unpaper_all_deps,
        install : false,
    )

    test('cuda deskew', cuda_deskew_test)

    cuda_stream_test = executable(
        'cuda_stream_test',
        [
            'tests/cuda_stream_test.c',
            'imageprocess/cuda_runtime.c',
            'lib/logging.c',
        ],
        c_args : unpaper_c_args,
        dependencies : unpaper_deps,
        install : false,
    )

    test('cuda stream', cuda_stream_test)

    opencv_bridge_test = executable(
        'opencv_bridge_test',
        [
            'tests/opencv_bridge_test.c',
            'imageprocess/cuda_runtime.c',
            'imageprocess/opencv_bridge.cpp',
            'lib/logging.c',
        ],
        c_args : unpaper_c_args,
        cpp_args : cpp_enabled ? unpaper_cpp_args : [],
        dependencies : unpaper_all_deps,
        install : false,
    )

    test('opencv bridge', opencv_bridge_test)
endif
