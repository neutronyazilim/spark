#!/usr/bin/env bash
# SPDX-License-Identifier: MIT

set -Eeuo pipefail

readonly PROGRAM_NAME="spark"
readonly DEFAULT_REPOSITORY="https://github.com/neutronyazilim/spark.git"

if [[ -z "${HOME:-}" ]]; then
  printf 'spark installer: HOME is not set\n' >&2
  exit 1
fi

prefix="${SPARK_INSTALL_PREFIX:-$HOME/.local}"
repository="${SPARK_REPOSITORY_URL:-$DEFAULT_REPOSITORY}"
modify_path=true
temporary_dir=""

usage() {
  cat <<EOF
Install spark from source.

Usage: ./install.sh [options]

Options:
  --prefix DIR       Install to DIR/bin (default: $HOME/.local/bin)
  --system           Install to /usr/local/bin
  --repo URL         Repository to clone when run outside a source checkout
  --no-modify-path   Do not update the current user's shell profile
  -h, --help         Show this help message

Environment:
  SPARK_INSTALL_PREFIX   Default installation prefix
  SPARK_REPOSITORY_URL   Default source repository
EOF
}

log() {
  printf '[spark] %s\n' "$*"
}

warn() {
  printf '[spark] warning: %s\n' "$*" >&2
}

die() {
  printf '[spark] error: %s\n' "$*" >&2
  exit 1
}

require_value() {
  local option="$1"
  local value="${2:-}"
  [[ -n "$value" ]] || die "$option requires a value"
}

while (($# > 0)); do
  case "$1" in
    --prefix)
      require_value "$1" "${2:-}"
      prefix="$2"
      shift 2
      ;;
    --prefix=*)
      prefix="${1#*=}"
      [[ -n "$prefix" ]] || die "--prefix requires a value"
      shift
      ;;
    --system)
      prefix="/usr/local"
      shift
      ;;
    --repo)
      require_value "$1" "${2:-}"
      repository="$2"
      shift 2
      ;;
    --repo=*)
      repository="${1#*=}"
      [[ -n "$repository" ]] || die "--repo requires a value"
      shift
      ;;
    --no-modify-path)
      modify_path=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1 (use --help for usage)"
      ;;
  esac
done

[[ "$prefix" == /* ]] || die "installation prefix must be an absolute path"
if [[ "$prefix" != "/" ]]; then
  prefix="${prefix%/}"
fi

if [[ "$prefix" == "/" ]]; then
  bin_dir="/bin"
else
  bin_dir="$prefix/bin"
fi
install_target="$bin_dir/$PROGRAM_NAME"

cleanup() {
  if [[ -n "$temporary_dir" && -d "$temporary_dir" ]]; then
    rm -rf -- "$temporary_dir"
  fi
}
trap cleanup EXIT

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
if [[ -f "$script_dir/Cargo.toml" && -f "$script_dir/src/main.rs" ]]; then
  source_dir="$script_dir"
else
  command -v git >/dev/null 2>&1 || die "git is required to download the source"
  temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/spark-install.XXXXXX")"
  source_dir="$temporary_dir/spark"
  log "Cloning $repository"
  git clone --depth 1 "$repository" "$source_dir"
fi

command -v cargo >/dev/null 2>&1 ||
  die "Cargo is required (on Arch Linux, install it with: sudo pacman -S rust)"
command -v install >/dev/null 2>&1 || die "the install utility is required"

log "Building an optimized binary"
(
  cd "$source_dir"
  CARGO_TARGET_DIR="$source_dir/target" cargo build --release --locked
)

built_binary="$source_dir/target/release/$PROGRAM_NAME"
[[ -x "$built_binary" ]] || die "build completed without producing $built_binary"

install_without_privileges() {
  mkdir -p "$bin_dir" 2>/dev/null &&
    [[ -w "$bin_dir" ]] &&
    install -m 0755 "$built_binary" "$install_target"
}

if ! install_without_privileges; then
  command -v sudo >/dev/null 2>&1 ||
    die "cannot write to $bin_dir; choose a user-writable --prefix or install sudo"
  log "Administrator access is required to install to $bin_dir"
  sudo install -Dm0755 "$built_binary" "$install_target"
fi

[[ -x "$install_target" ]] || die "failed to install $install_target"
log "Installed $install_target"

path_contains() {
  case ":${PATH:-}:" in
    *":$bin_dir:"*) return 0 ;;
    *) return 1 ;;
  esac
}

profile_path_line=""
profile_file=""

select_profile() {
  local configured_shell="${SHELL:-}"
  local shell_name="${configured_shell##*/}"
  local escaped_path

  case "$shell_name" in
    fish)
      profile_file="$HOME/.config/fish/config.fish"
      escaped_path="${bin_dir//\\/\\\\}"
      escaped_path="${escaped_path//\"/\\\"}"
      escaped_path="${escaped_path//\$/\\\$}"
      profile_path_line="fish_add_path --path \"$escaped_path\""
      ;;
    zsh)
      profile_file="$HOME/.zshrc"
      printf -v escaped_path '%q' "$bin_dir"
      profile_path_line="export PATH=$escaped_path:\$PATH"
      ;;
    bash)
      profile_file="$HOME/.bashrc"
      printf -v escaped_path '%q' "$bin_dir"
      profile_path_line="export PATH=$escaped_path:\$PATH"
      ;;
    *)
      profile_file="$HOME/.profile"
      printf -v escaped_path '%q' "$bin_dir"
      profile_path_line="export PATH=$escaped_path:\$PATH"
      ;;
  esac
}

add_to_profile() {
  select_profile
  if ! mkdir -p "$(dirname -- "$profile_file")" 2>/dev/null ||
    ! touch "$profile_file" 2>/dev/null; then
    warn "could not update $profile_file"
    return 1
  fi

  if grep -Fqx "$profile_path_line" "$profile_file" 2>/dev/null; then
    return 0
  fi

  if ! printf '\n# Added by the spark installer.\n%s\n' "$profile_path_line" >>"$profile_file"; then
    warn "could not update $profile_file"
    return 1
  fi
  log "Added $bin_dir to PATH in $profile_file"
}

if ! path_contains; then
  if [[ "$modify_path" == true ]] && add_to_profile; then
    log "Restart your shell to use '$PROGRAM_NAME', or run:"
  else
    warn "$bin_dir is not currently on PATH; run:"
  fi
  printf '  export PATH=%q:$PATH\n' "$bin_dir"
else
  log "Run '$PROGRAM_NAME' from any directory"
fi
