"""Configure/build the pinned R10 Windows library and Vulkan plugin together.

Run from the experiment repo. The build-only workarounds target the extracted
source under outputs/r10. No archived runtime DLL is copied into this build.
"""

import argparse
from hashlib import file_digest
from pathlib import Path
import subprocess

SOURCE = Path("outputs/r10/upstream/llama.cpp-8a56aedd6")
BUILD = Path("outputs/r10/llama-native-vulkan-build")
CMAKE = Path("outputs/r10/tools/cmake-3.31.6-windows-x86_64/bin/cmake.exe")
SDK = Path("outputs/r10/tools/msys/ucrt64")
ARCHIVES = {
    "shaderc-2026.4-1.tar.zst": "a4ad271f76acd2307ed67738770b30d6486bd3c7acff63ba83dd13e56ce9ab82",
    "vulkan-loader-1~1.4.363.0-1.tar.zst": "065076fe13d39ab97eea4740596426d5ae741b5044e48e1c9495d04835c92dbe",
    "spirv-headers-2~1.4.363.0-1.tar.zst": "c15f7df169ccd4bf57fc9053d34e762818fa96e3da229c92501d7eedf4aa71dd",
    "vulkan-headers-1~1.4.363.0-1.tar.zst": "be602b2657cace531316e78b1eda4087e8f7f84ebb877f08fba198a2db06406a",
}


VULKAN_BASE_FLAGS = "-O0 -fno-if-conversion -fno-if-conversion2 -fno-thread-jumps"
VULKAN_DIRECT_FLAGS = "-O1 -fno-inline -fno-jump-tables -fno-if-conversion -fno-if-conversion2 -fno-thread-jumps"


def check_paths_and_patch(direct):
    for name, expected in ARCHIVES.items():
        with (SOURCE.parent / name).open("rb") as f:
            if file_digest(f, "sha256").hexdigest() != expected:
                raise ValueError(f"SDK input hash mismatch: {name}")
    path = SOURCE / "ggml/src/ggml-vulkan/CMakeLists.txt"
    before = path.read_text(encoding="utf-8")
    target_option = "target_compile_options(ggml-vulkan PRIVATE {})"
    after = before.replace(target_option.format(VULKAN_DIRECT_FLAGS + " -fno-forward-propagate"),
                           target_option.format(VULKAN_DIRECT_FLAGS))
    after = after.replace(target_option.format(VULKAN_BASE_FLAGS if direct else VULKAN_DIRECT_FLAGS),
                          target_option.format(VULKAN_DIRECT_FLAGS if direct else VULKAN_BASE_FLAGS))
    for anchor, added in (
        ('        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vulkan-shaders\n',
         '        BINARY_DIR "${CMAKE_BINARY_DIR}/r10_shader_gen"\n'),
        ('                   ${VULKAN_SHADER_GEN_CMAKE_ARGS}\n',
         '                   "-DCMAKE_CXX_FLAGS_RELEASE:STRING=-O0 -DNDEBUG -fno-if-conversion -fno-if-conversion2 -fno-thread-jumps"\n'),
        ('    target_link_libraries(ggml-vulkan PRIVATE Vulkan::Vulkan)\n',
         '    if (MINGW AND CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL "14.1.0")\n'
         f'        {target_option.format(VULKAN_DIRECT_FLAGS if direct else VULKAN_BASE_FLAGS)}\n'
         '    endif()\n'),
    ):
        if after.count(anchor) != 1:
            raise ValueError(f"Unexpected version or edited source around {anchor.strip()}")
        if added + anchor not in after:
            after = after.replace(anchor, added + anchor)
    if after != before:
        path.write_text(after, encoding="utf-8", newline="\n")


def rebuild_direct_objects(jobs, toolchain_fix=False):
    # The pinned MinGW crashes when relinking CMake's whole-archive on this
    # host. Link the *same* compiled input objects, using CMake's response list.
    # Unchanged core/model/shader units come from the same-source base build.
    work = []
    if toolchain_fix:
        work.append(("ggml/src", "ggml-base", ["ggml.c.obj"], "../../bin/ggml-base.dll", 23,
                     ["-Wl,--out-implib,libggml-base.dll.a"]))
    else:
        work.append(("ggml/src/ggml-vulkan", "ggml-vulkan", ["ggml-vulkan.cpp.obj"], "../../../bin/ggml-vulkan.dll", 23, []))
    core = ["llama-context.cpp.obj", "llama-impl.cpp.obj"] if toolchain_fix else ["llama-context.cpp.obj"]
    work.append(("src", "llama", core, "../bin/libllama.dll", 4, ["-Wl,--out-implib,libllama.dll.a"]))
    for subdir, target, objects, out, ver, extra in work:
        rules = f"CMakeFiles/{target}.dir"
        rel = f"{subdir}/{rules}"
        for prerequisite in ("objects1.rsp", "linkLibs.rsp", "build.make"):
            if not (BUILD / rel / prerequisite).is_file():
                raise FileNotFoundError(f"Build the same-source baseline first: {rel}/{prerequisite}")
        subprocess.run(["mingw32-make.exe", f"-j{jobs}", "-f", f"{rel}/build.make", *(f"{rel}/{obj}" for obj in objects)],
                       cwd=BUILD, check=True)
        subprocess.run(["g++.exe", "-shared", "-o", out, *extra, f"-Wl,--major-image-version,0,--minor-image-version,{ver}",
                        f"@{rules}/objects1.rsp", f"@{rules}/linkLibs.rsp"], cwd=BUILD / subdir, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configure-only", action="store_true")
    p.add_argument("--direct", action="store_true", help="rebuild the patched restore and Vulkan units after the baseline")
    p.add_argument("--format-fix", action="store_true", help="relink core format units after a same-source direct build")
    p.add_argument("--jobs", type=int, default=2)
    args = p.parse_args()
    if args.jobs < 1:
        p.error("--jobs must be positive")
    if args.format_fix and (not args.direct or not (SOURCE / "ggml/src/r10-win-format.h").is_file() or
                            not (BUILD / "bin/ggml-vulkan.dll").is_file()):
        p.error("--format-fix needs an existing --direct build and the formatting patch")
    if args.direct:
        for path, symbol in ((SOURCE / "src/llama-context.cpp", "llama_r10_prepare_seq_restore"),
                             (SOURCE / "ggml/src/ggml-vulkan/ggml-vulkan.cpp", "ggml_backend_vk_r10_submit")):
            if symbol not in path.read_text(encoding="utf-8"):
                p.error("apply the two R10 direct-restore patches first")
    check_paths_and_patch(args.direct)
    sdk = SDK.resolve(strict=True).as_posix()
    cmake = str(CMAKE.resolve(strict=True))
    cmd = [
        cmake, "-S", str(SOURCE), "-B", str(BUILD), "-G", "MinGW Makefiles",
        "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON", "-DLLAMA_BUILD_COMMON=OFF",
        "-DLLAMA_BUILD_TOOLS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_CURL=OFF", "-DGGML_VULKAN=ON", "-DGGML_BACKEND_DL=ON",
        "-DGGML_NATIVE=OFF", "-DGGML_OPENMP=OFF", "-DGGML_CCACHE=OFF",
        "-DCMAKE_CXX_FLAGS_RELEASE=-O0 -DNDEBUG -fno-if-conversion -fno-if-conversion2 -fno-thread-jumps",
        "-DCMAKE_C_FLAGS_RELEASE=-O0 -DNDEBUG -fno-if-conversion -fno-if-conversion2 -fno-thread-jumps",
        "-DCMAKE_SHARED_LINKER_FLAGS=",
        f"-DCMAKE_PREFIX_PATH={sdk}",
        f"-DVulkan_INCLUDE_DIR={sdk}/include", f"-DVulkan_LIBRARY={sdk}/lib/libvulkan-1.dll.a",
        f"-DVulkan_GLSLC_EXECUTABLE={sdk}/bin/glslc.exe",
    ]
    subprocess.run(cmd, check=True)
    if args.configure_only:
        return
    if args.direct:
        rebuild_direct_objects(args.jobs, toolchain_fix=args.format_fix)
    else:
        subprocess.run([cmake, "--build", str(BUILD), "--target", "llama", "ggml-cpu", "ggml-vulkan",
                        "--parallel", str(args.jobs)], check=True)


if __name__ == "__main__":
    main()
