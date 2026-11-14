pkgname=spark-llm-git
_pkgname=spark
pkgver=0.1.0.r5.g273fe8e
pkgrel=1
pkgdesc='CPU-first GGUF LLM inference engine with optional CUDA acceleration'
arch=('x86_64')
url='https://github.com/neutronyazilim/spark'
license=('MIT')
depends=('glibc' 'libgcc')
makedepends=('cargo' 'git')
provides=("spark-llm=${pkgver}")
conflicts=('spark' 'spark-llm')
source=("${_pkgname}::git+${url}.git")
sha256sums=('SKIP')

pkgver() {
  cd "$_pkgname"
  local _version
  _version="$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -n 1)"
  printf '%s.r%s.g%s\n' \
    "$_version" \
    "$(git rev-list --count HEAD)" \
    "$(git rev-parse --short=7 HEAD)"
}

prepare() {
  cd "$_pkgname"
  export RUSTUP_TOOLCHAIN=stable
  cargo fetch --locked --target "$CARCH-unknown-linux-gnu"
}

build() {
  cd "$_pkgname"
  export RUSTUP_TOOLCHAIN=stable
  export CARGO_TARGET_DIR=target
  cargo build --frozen --release
}

check() {
  cd "$_pkgname"
  export RUSTUP_TOOLCHAIN=stable
  export CARGO_TARGET_DIR=target
  cargo test --frozen
}

package() {
  cd "$_pkgname"
  install -Dm0755 target/release/spark "$pkgdir/usr/bin/spark"
  install -Dm0644 README.md "$pkgdir/usr/share/doc/$pkgname/README.md"
  install -Dm0644 LICENSE "$pkgdir/usr/share/licenses/$pkgname/LICENSE"
}
