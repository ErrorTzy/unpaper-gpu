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
    dependency('libswscale'),
    cc.find_library('m', required : false),
    dependency('threads')
]

# PDF support via MuPDF
pdf_opt = get_option('pdf')
have_pdf = false

if pdf_opt.enabled() or pdf_opt.auto()
    mupdf_dep = dependency('mupdf', required : pdf_opt.enabled())
    if mupdf_dep.found()
        have_pdf = true
        unpaper_deps += [mupdf_dep]
    endif
endif

conf_data = configuration_data()
conf_data.set('version', meson.project_version())
configure_file(input: 'version.h.in', output: 'version.h', configuration: conf_data)

cuda_opt = get_option('cuda')
have_cuda = false
nvcc = disabler()
unpaper_c_args = []
unpaper_cpp_args = []
unpaper_sources = [
    'file.c', 'parse.c', 'sheet_process.c',
    'src/cli/cli_main.c', 'src/cli/cli_options.c',
    'src/core/sheet_pipeline.c',
    'src/core/sheet_stages.c',
    'imageprocess/blit.c',
    'imageprocess/backend.c',
    'imageprocess/cuda_mempool.c',
    'imageprocess/cuda_stream_pool.c',
    'imageprocess/deskew.c',
    'imageprocess/interpolate.c',
    'imageprocess/fill.c',
    'imageprocess/filters.c',
    'imageprocess/image.c',
    'imageprocess/image_cuda.c',
    'imageprocess/masks.c',
    'imageprocess/pixel.c',
    'imageprocess/primitives.c',
    'lib/batch.c',
    'lib/batch_decode_queue.c',
    'lib/batch_worker.c',
    'lib/decode_queue.c',
    'lib/encode_queue.c',
    'lib/gpu_monitor.c',
    'lib/logging.c',
    'lib/perf.c',
    'lib/options.c',
    'lib/physical.c',
    'lib/threadpool.c',
]

# PDF support sources
if have_pdf
    unpaper_c_args += ['-DUNPAPER_WITH_PDF=1']
    unpaper_sources += [
        'pdf/pdf_reader.c',
        'pdf/pdf_writer.c',
        'pdf/pdf_pipeline_cpu.c',
        'pdf/pdf_pipeline_cpu_batch.c',
        'pdf/pdf_pipeline_decode.c',
        'pdf/pdf_page_accumulator.c',
        'pdf/pdf_pipeline_batch.c',
    ]
endif

# GPU PDF pipeline requires both PDF and CUDA
# This is added after CUDA detection (line 234) is complete

# JBIG2 support via jbig2dec (for B&W PDF images)
jbig2_opt = get_option('jbig2')
have_jbig2 = false

if jbig2_opt.enabled() or jbig2_opt.auto()
    jbig2dec_dep = cc.find_library('jbig2dec', required : jbig2_opt.enabled())
    if jbig2dec_dep.found()
        # Check for header
        if cc.has_header('jbig2.h')
            have_jbig2 = true
            unpaper_c_args += ['-DUNPAPER_WITH_JBIG2=1']
            unpaper_deps += [jbig2dec_dep]
            unpaper_sources += ['lib/jbig2_decode.c']
        elif jbig2_opt.enabled()
            error('jbig2dec library found but jbig2.h header not found')
        endif
    endif
endif

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

# CUDA stub for non-CUDA builds (must be after CUDA detection)
if not have_cuda
    unpaper_sources += [
        'imageprocess/backend_cuda_stub.c',
    ]
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
        'imageprocess/backend_cuda_blit.c',
        'imageprocess/backend_cuda_masks.c',
        'imageprocess/backend_cuda_filters.c',
        'imageprocess/backend_cuda_deskew.c',
        'imageprocess/cuda_runtime.c',
        'imageprocess/npp_wrapper.c',
        'imageprocess/npp_integral.c',
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

    # Link NPP libraries for GPU integral image computation
    nppc_dep = cc.find_library('nppc', dirs : [
        '/usr/local/cuda-13.0/lib64',
        '/usr/local/cuda/lib64',
        '/usr/lib/x86_64-linux-gnu'
    ], required : true)
    nppist_dep = cc.find_library('nppist', dirs : [
        '/usr/local/cuda-13.0/lib64',
        '/usr/local/cuda/lib64',
        '/usr/lib/x86_64-linux-gnu'
    ], required : true)
    nppicc_dep = cc.find_library('nppicc', dirs : [
        '/usr/local/cuda-13.0/lib64',
        '/usr/local/cuda/lib64',
        '/usr/lib/x86_64-linux-gnu'
    ], required : true)
    unpaper_deps += [nppc_dep, nppist_dep, nppicc_dep]

    # Link nvJPEG library for GPU JPEG decode/encode
    # nvJPEG is part of CUDA Toolkit, no separate install needed
    nvjpeg_dep = cc.find_library('nvjpeg', dirs : [
        '/usr/local/cuda-13.0/lib64',
        '/usr/local/cuda/lib64',
        '/usr/lib/x86_64-linux-gnu'
    ], required : true)
    unpaper_deps += [nvjpeg_dep]

    # nvImageCodec for unified JPEG/JPEG2000 GPU decode/encode
    # Required for CUDA builds - provides GPU decode/encode via nvjpeg_ext plugin
    nvimgcodec_dep = cc.find_library('nvimgcodec', dirs : [
        '/usr/local/cuda-13.0/lib64',
        '/usr/local/cuda/lib64',
        '/opt/nvidia/nvimgcodec_cuda13/lib64',
        '/opt/nvidia/nvimgcodec_cuda12/lib64',
        '/opt/nvidia/nvimgcodec/lib',
        '/usr/lib/x86_64-linux-gnu'
    ], required : true)
    unpaper_c_args += ['-DUNPAPER_WITH_NVIMGCODEC=1']
    unpaper_deps += [nvimgcodec_dep]

    # Add nvImageCodec include path
    unpaper_c_args += ['-I/opt/nvidia/nvimgcodec_cuda13/include', '-I/opt/nvidia/nvimgcodec_cuda12/include']

    # Also need nvjpeg2k for JPEG2000 support (optional)
    nvjpeg2k_dep = cc.find_library('nvjpeg2k', dirs : [
        '/usr/local/cuda-13.0/lib64',
        '/usr/local/cuda/lib64',
        '/opt/nvidia/nvjpeg2k_cuda13/lib64',
        '/opt/nvidia/nvjpeg2k_cuda12/lib64',
        '/opt/nvidia/nvjpeg2k/lib',
        '/usr/lib/x86_64-linux-gnu'
    ], required : false)
    if nvjpeg2k_dep.found()
        unpaper_deps += [nvjpeg2k_dep]
    endif

    # nvimgcodec wrapper source
    unpaper_sources += ['imageprocess/nvimgcodec.c']

    # GPU PDF pipeline (requires both CUDA and PDF)
    if have_pdf
        unpaper_sources += ['pdf/pdf_pipeline_gpu.c']
    endif
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

cli_options_test = executable(
    'cli_options_test',
    [
        'tests/cli_options_test.c',
        'parse.c',
        'lib/options.c',
        'lib/logging.c',
        'imageprocess/primitives.c',
    ],
    c_args : unpaper_c_args,
    dependencies : unpaper_deps,
    install : false,
)

test('cli options', cli_options_test)

sheet_stages_test_sources = [
    'tests/sheet_stages_test.c',
    'file.c',
    'parse.c',
    'sheet_process.c',
    'src/core/sheet_stages.c',
    'imageprocess/backend.c',
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
    'lib/encode_queue.c',
    'lib/logging.c',
    'lib/options.c',
    'lib/perf.c',
]

if have_cuda
    sheet_stages_test_sources += [
        'imageprocess/backend_cuda.c',
        'imageprocess/backend_cuda_blit.c',
        'imageprocess/backend_cuda_masks.c',
        'imageprocess/backend_cuda_filters.c',
        'imageprocess/backend_cuda_deskew.c',
        'imageprocess/cuda_mempool.c',
        'imageprocess/cuda_runtime.c',
        'imageprocess/npp_wrapper.c',
        'imageprocess/npp_integral.c',
        'imageprocess/nvimgcodec.c',
        'imageprocess/opencv_bridge.cpp',
        'imageprocess/opencv_ops.cpp',
        cuda_kernels_ptx_c,
    ]
else
    sheet_stages_test_sources += ['imageprocess/backend_cuda_stub.c']
endif

sheet_stages_test = executable(
    'sheet_stages_test',
    sheet_stages_test_sources,
    c_args : unpaper_c_args,
    cpp_args : cpp_enabled ? unpaper_cpp_args : [],
    dependencies : unpaper_all_deps,
    install : false,
)

test('sheet stages', sheet_stages_test, is_parallel: false)

if have_cuda
    cuda_roundtrip_test = executable(
        'cuda_roundtrip_test',
        [
            'tests/cuda_roundtrip_test.c',
            'imageprocess/cuda_mempool.c',
            'imageprocess/image_cuda.c',
            'imageprocess/cuda_runtime.c',
            'lib/logging.c',
        ],
        c_args : unpaper_c_args,
        dependencies : unpaper_deps,
        install : false,
    )

    # cuda_roundtrip_test doesn't switch backends, can run in parallel
    test('cuda roundtrip', cuda_roundtrip_test)

    cuda_primitives_test = executable(
        'cuda_primitives_test',
        [
            'tests/cuda_primitives_test.c',
            'imageprocess/backend.c',
            'imageprocess/backend_cuda.c',
            'imageprocess/backend_cuda_blit.c',
            'imageprocess/backend_cuda_masks.c',
            'imageprocess/backend_cuda_filters.c',
            'imageprocess/backend_cuda_deskew.c',
            'imageprocess/blit.c',
            'imageprocess/cuda_mempool.c',
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
            'imageprocess/npp_wrapper.c',
            'imageprocess/npp_integral.c',
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

    test('cuda primitives', cuda_primitives_test, is_parallel: false)

    cuda_resize_test = executable(
        'cuda_resize_test',
        [
            'tests/cuda_resize_test.c',
            'imageprocess/backend.c',
            'imageprocess/backend_cuda.c',
            'imageprocess/backend_cuda_blit.c',
            'imageprocess/backend_cuda_masks.c',
            'imageprocess/backend_cuda_filters.c',
            'imageprocess/backend_cuda_deskew.c',
            'imageprocess/blit.c',
            'imageprocess/cuda_mempool.c',
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
            'imageprocess/npp_wrapper.c',
            'imageprocess/npp_integral.c',
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

    test('cuda resize', cuda_resize_test, is_parallel: false)

    cuda_filters_test = executable(
        'cuda_filters_test',
        [
            'tests/cuda_filters_test.c',
            'imageprocess/backend.c',
            'imageprocess/backend_cuda.c',
            'imageprocess/backend_cuda_blit.c',
            'imageprocess/backend_cuda_masks.c',
            'imageprocess/backend_cuda_filters.c',
            'imageprocess/backend_cuda_deskew.c',
            'imageprocess/blit.c',
            'imageprocess/cuda_mempool.c',
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
            'imageprocess/npp_wrapper.c',
            'imageprocess/npp_integral.c',
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

    test('cuda filters', cuda_filters_test, is_parallel: false)

    cuda_masks_border_test = executable(
        'cuda_masks_border_test',
        [
            'tests/cuda_masks_border_test.c',
            'imageprocess/backend.c',
            'imageprocess/backend_cuda.c',
            'imageprocess/backend_cuda_blit.c',
            'imageprocess/backend_cuda_masks.c',
            'imageprocess/backend_cuda_filters.c',
            'imageprocess/backend_cuda_deskew.c',
            'imageprocess/blit.c',
            'imageprocess/cuda_mempool.c',
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
            'imageprocess/npp_wrapper.c',
            'imageprocess/npp_integral.c',
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

    test('cuda masks/borders', cuda_masks_border_test, is_parallel: false)

    cuda_deskew_test = executable(
        'cuda_deskew_test',
        [
            'tests/cuda_deskew_test.c',
            'imageprocess/backend.c',
            'imageprocess/backend_cuda.c',
            'imageprocess/backend_cuda_blit.c',
            'imageprocess/backend_cuda_masks.c',
            'imageprocess/backend_cuda_filters.c',
            'imageprocess/backend_cuda_deskew.c',
            'imageprocess/blit.c',
            'imageprocess/cuda_mempool.c',
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
            'imageprocess/npp_wrapper.c',
            'imageprocess/npp_integral.c',
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

    test('cuda deskew', cuda_deskew_test, is_parallel: false)

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

    # cuda_stream_test doesn't switch backends, can run in parallel
    test('cuda stream', cuda_stream_test)

    opencv_bridge_test = executable(
        'opencv_bridge_test',
        [
            'tests/opencv_bridge_test.c',
            'imageprocess/cuda_runtime.c',
            'imageprocess/cuda_mempool.c',
            'imageprocess/npp_wrapper.c',
            'imageprocess/npp_integral.c',
            'imageprocess/opencv_bridge.cpp',
            cuda_kernels_ptx_c,
            'lib/logging.c',
        ],
        c_args : unpaper_c_args,
        cpp_args : cpp_enabled ? unpaper_cpp_args : [],
        dependencies : unpaper_all_deps,
        install : false,
    )

    test('opencv bridge', opencv_bridge_test)

    npp_integral_test = executable(
        'npp_integral_test',
        [
            'tests/npp_integral_test.c',
            'imageprocess/cuda_runtime.c',
            'imageprocess/cuda_mempool.c',
            'imageprocess/npp_wrapper.c',
            'imageprocess/npp_integral.c',
            'lib/logging.c',
        ],
        c_args : unpaper_c_args,
        dependencies : unpaper_deps,
        install : false,
    )

    # npp_integral_test doesn't switch backends, can run in parallel
    test('npp integral', npp_integral_test)

    cuda_blurfilter_scan_test = executable(
        'cuda_blurfilter_scan_test',
        [
            'tests/cuda_blurfilter_scan_test.c',
            'imageprocess/cuda_runtime.c',
            'imageprocess/cuda_mempool.c',
            'imageprocess/npp_wrapper.c',
            'imageprocess/npp_integral.c',
            cuda_kernels_ptx_c,
            'lib/logging.c',
        ],
        c_args : unpaper_c_args,
        dependencies : unpaper_deps,
        install : false,
    )

    # cuda_blurfilter_scan_test doesn't switch backends, can run in parallel
    test('cuda blurfilter scan', cuda_blurfilter_scan_test)

    cuda_grayfilter_scan_test = executable(
        'cuda_grayfilter_scan_test',
        [
            'tests/cuda_grayfilter_scan_test.c',
            'imageprocess/cuda_runtime.c',
            'imageprocess/cuda_mempool.c',
            'imageprocess/npp_wrapper.c',
            'imageprocess/npp_integral.c',
            cuda_kernels_ptx_c,
            'lib/logging.c',
        ],
        c_args : unpaper_c_args,
        dependencies : unpaper_deps,
        install : false,
    )

    # cuda_grayfilter_scan_test doesn't switch backends, can run in parallel
    test('cuda grayfilter scan', cuda_grayfilter_scan_test)

    nvimgcodec_test = executable(
        'nvimgcodec_test',
        [
            'tests/nvimgcodec_test.c',
            'imageprocess/cuda_runtime.c',
            'imageprocess/nvimgcodec.c',
            'lib/logging.c',
        ],
        c_args : unpaper_c_args,
        dependencies : unpaper_deps,
        install : false,
    )

    # nvimgcodec_test tests the unified GPU codec wrapper
    test('nvimgcodec wrapper', nvimgcodec_test,
         env: ['TEST_IMGSRC_DIR=' + meson.project_source_root() + '/tests/source_images/',
               'CUDA_LAUNCH_BLOCKING=1'])

    # gpu jpeg pipeline test runs the unpaper binary, can run in parallel
    test(
        'gpu jpeg pipeline',
        python,
        args: [
            '-m', 'pytest',
            '--basetemp=' + meson.current_build_dir() + '/test_output_gpu/',
            meson.project_source_root() + '/tests/gpu_jpeg_pipeline_tests.py'
        ],
        env: [
            'TEST_IMGSRC_DIR=' + meson.project_source_root() + '/tests/source_images/',
            'TEST_GOLDEN_DIR=' + meson.project_source_root() + '/tests/golden_images/',
            'TEST_UNPAPER_BINARY=' + unpaper.full_path(),
        ],
        timeout : -1,
    )
endif

# PDF support tests
if have_pdf
    pdf_reader_test = executable(
        'pdf_reader_test',
        [
            'tests/pdf_reader_test.c',
            'pdf/pdf_reader.c',
        ],
        c_args : unpaper_c_args,
        dependencies : unpaper_deps,
        install : false,
    )

    test('pdf reader', pdf_reader_test,
         env: ['TEST_IMGSRC_DIR=' + meson.project_source_root() + '/tests/source_images/'])

    pdf_writer_test = executable(
        'pdf_writer_test',
        [
            'tests/pdf_writer_test.c',
            'pdf/pdf_reader.c',
            'pdf/pdf_writer.c',
        ],
        c_args : unpaper_c_args,
        dependencies : unpaper_deps,
        install : false,
    )

    test('pdf writer', pdf_writer_test,
         env: ['TEST_IMGSRC_DIR=' + meson.project_source_root() + '/tests/source_images/'])

    # PDF pipeline CPU test - uses processing pipeline without main()
    pdf_pipeline_cpu_test_sources = [
        'file.c', 'parse.c', 'sheet_process.c',
        'src/core/sheet_pipeline.c',
        'imageprocess/blit.c',
        'imageprocess/backend.c',
        'imageprocess/cuda_mempool.c',
        'imageprocess/cuda_stream_pool.c',
        'imageprocess/deskew.c',
        'imageprocess/interpolate.c',
        'imageprocess/fill.c',
        'imageprocess/filters.c',
        'imageprocess/image.c',
        'imageprocess/image_cuda.c',
        'imageprocess/masks.c',
        'imageprocess/pixel.c',
        'imageprocess/primitives.c',
        'lib/batch.c',
        'lib/batch_decode_queue.c',
        'lib/batch_worker.c',
        'lib/decode_queue.c',
        'lib/encode_queue.c',
        'lib/gpu_monitor.c',
        'lib/logging.c',
        'lib/perf.c',
        'lib/options.c',
        'lib/physical.c',
        'lib/threadpool.c',
        'pdf/pdf_reader.c',
        'pdf/pdf_writer.c',
        'pdf/pdf_pipeline_cpu.c',
        'pdf/pdf_pipeline_cpu_batch.c',
        'pdf/pdf_pipeline_decode.c',
        'pdf/pdf_page_accumulator.c',
        'pdf/pdf_pipeline_batch.c',
        'tests/pdf_pipeline_cpu_test.c',
    ]
    if have_cuda
        pdf_pipeline_cpu_test_sources += [
            'imageprocess/backend_cuda.c',
            'imageprocess/backend_cuda_blit.c',
            'imageprocess/backend_cuda_masks.c',
            'imageprocess/backend_cuda_filters.c',
            'imageprocess/backend_cuda_deskew.c',
            'imageprocess/cuda_runtime.c',
            'imageprocess/npp_wrapper.c',
            'imageprocess/npp_integral.c',
            'imageprocess/nvimgcodec.c',
            'imageprocess/opencv_bridge.cpp',
            'imageprocess/opencv_ops.cpp',
            cuda_kernels_ptx_c,
        ]
    else
        pdf_pipeline_cpu_test_sources += ['imageprocess/backend_cuda_stub.c']
    endif
    if have_jbig2
        pdf_pipeline_cpu_test_sources += ['lib/jbig2_decode.c']
    endif
    pdf_pipeline_cpu_test = executable(
        'pdf_pipeline_cpu_test',
        pdf_pipeline_cpu_test_sources,
        c_args : unpaper_c_args,
        cpp_args : cpp_enabled ? unpaper_cpp_args : [],
        dependencies : unpaper_all_deps,
        install : false,
    )

    test('pdf pipeline cpu', pdf_pipeline_cpu_test,
         env: ['TEST_IMGSRC_DIR=' + meson.project_source_root() + '/tests/source_images/'],
         timeout: 300)

    # JBIG2 decode test - requires PDF for full integration tests
    if have_jbig2
        jbig2_decode_test = executable(
            'jbig2_decode_test',
            [
                'tests/jbig2_decode_test.c',
                'lib/jbig2_decode.c',
                'pdf/pdf_reader.c',
            ],
            c_args : unpaper_c_args,
            dependencies : unpaper_deps,
            install : false,
        )

        test('jbig2 decode', jbig2_decode_test,
             env: ['TEST_IMGSRC_DIR=' + meson.project_source_root() + '/tests/source_images/'])
    endif
endif
