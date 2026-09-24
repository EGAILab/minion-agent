#!/usr/bin/env bash
# Build the ONE pinned ICU that `ls` collation (R006-C, spec/tools.md TOOL-028) requires, and print
# the environment that `uv sync` (PyICU 2.16.2 from its lock-hashed sdist) and the runtime need.
#
#   bash scripts/pinned-icu/build.sh <prefix>        # e.g. ../.toolchain/icu-78.3
#   eval "$(bash scripts/pinned-icu/build.sh <prefix> --env)"
#
# Windows: Git Bash + Visual Studio 2022 (MSBuild found via vswhere). Linux: a C/C++ toolchain.
# Nothing is installed outside <prefix>.
set -euo pipefail
PREFIX=$(mkdir -p "$1" && cd "$1" && pwd)
MODE=${2:-build}
REL=https://github.com/unicode-org/icu/releases/download/release-78.3
TGZ=icu4c-78.3-sources.tgz
SHA512=04a49455e1489030c520a4bfd2664fa2171e7938d08f2acdbbcb1fda976639fd8b1f0704f2eec89ba59a7b6d118ceaab6ec5a096e40d9085a0895d91ce225245

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) OS=windows ;;
  Linux) OS=linux ;;
  *) echo "unsupported platform: $(uname -s)" >&2; exit 1 ;;
esac

print_env() {
  if [ "$OS" = windows ]; then
    local w; w=$(cygpath -w "$PREFIX/icu")
    # Libraries by ABSOLUTE path: the Windows SDK ships its own icuuc.lib/icuin.Lib (the system
    # ICU) on the default library path, and a bare-name link silently picks those up.
    echo "export PYICU_INCLUDES='$w\\include'"
    echo "export PYICU_LFLAGS='/LIBPATH:$w\\lib64'"
    echo "export PYICU_LIBRARIES='$w\\lib64\\icuin;$w\\lib64\\icuuc;$w\\lib64\\icudt'"
    echo "export PYICU_CFLAGS='/Zc:wchar_t;/EHsc;/std:c++17'"
    echo "export MINION_AGENT_ICU_BIN='$w\\bin64'"
  else
    echo "export PYICU_INCLUDES='$PREFIX/install/include'"
    echo "export PYICU_LFLAGS='-L$PREFIX/install/lib:-Wl,-rpath,$PREFIX/install/lib'"
    # -L$PREFIX comes before the system library path, so bare names resolve to the pinned build.
    echo "export PYICU_LIBRARIES='icui18n:icuuc:icudata'"
    echo "export PYICU_CFLAGS='-std=c++17'"
  fi
  echo "export ICU_VERSION=78.3"
  echo "export UV_NO_CACHE=1  # rebuild PyICU against this ICU rather than reuse a cached wheel"
}

if [ "$MODE" = --env ]; then print_env; exit 0; fi

cd "$PREFIX"
[ -f "$TGZ" ] || curl -fsSLO "$REL/$TGZ"
echo "$SHA512 *$TGZ" | sha512sum -c -
rm -rf icu && tar xzf "$TGZ"
if [ "$OS" = windows ]; then
  VSWHERE="/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"
  MSBUILD=$("$VSWHERE" -latest -requires Microsoft.Component.MSBuild -find 'MSBuild\**\Bin\MSBuild.exe' | head -1)
  "$MSBUILD" icu/source/allinone/allinone.sln -p:Configuration=Release -p:Platform=x64 \
    -p:SkipUWP=true -m -v:minimal -nologo > build.log 2>&1
  test -f icu/bin64/icuuc78.dll
else
  (cd icu/source && ./runConfigureICU Linux --prefix="$PREFIX/install" --disable-samples \
    --disable-tests > "$PREFIX/configure.log" && make -j"$(nproc)" > "$PREFIX/make.log" 2>&1 \
    && make install > "$PREFIX/install.log" 2>&1)
  test -f install/lib/libicuuc.so.78
fi
echo "pinned ICU 78.3 built in $PREFIX"
print_env
