pkgname=spark-llm-git
pkgver=r1
pkgrel=1
pkgdesc='Minimal CPU-first GGUF LLM inference engine written in Rust'
arch=('x86_64')
url='https://github.com/deniz/spark'
license=('custom')
depends=('gcc-libs')
makedepends=('cargo' 'git')
provides=('spark')
conflicts=('spark')
source=('git+https://github.com/deniz/spark.git')
sha256sums=('SKIP')

pkgver() {
  cd "$srcdir/spark"
  printf 'r%s.%s\n' "$(git rev-list --count HEAD)" "$(git rev-parse --short HEAD)"
}

build() {
  cd "$srcdir/spark"
  cargo build --release --locked
}

package() {
  cd "$srcdir/spark"
  install -Dm755 target/release/spark "$pkgdir/usr/bin/spark"
  install -Dm644 README.md "$pkgdir/usr/share/doc/spark/README.md"
}
