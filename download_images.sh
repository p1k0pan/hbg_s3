#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${HF_REPO_ID:-p1k0/HBG}"
DEST="${1:-stage3_images}"
DOWNLOAD_DIR="${DEST}/downloads"
BASE_ARCHIVE="hbg_stage3_images.tar.zst"
BASE_CHECKSUM="hbg_stage3_images.tar.zst.sha256"
SUPPLEMENT_ARCHIVE="stage3_image_supplement_110.tar"
SUPPLEMENT_SHA256="5bd5ae0924229990dfc8b22b05682dd61947fb9f5c5db698a96adb0375a57ec0"

if ! command -v hf >/dev/null 2>&1; then
  printf '%s\n' \
    'Missing Hugging Face CLI.' \
    'Install dependencies with: python -m pip install -r requirements.txt' >&2
  exit 1
fi

mkdir -p "${DOWNLOAD_DIR}"
hf download \
  "${REPO_ID}" \
  "${BASE_ARCHIVE}" \
  "${BASE_CHECKSUM}" \
  "${SUPPLEMENT_ARCHIVE}" \
  --repo-type dataset \
  --local-dir "${DOWNLOAD_DIR}"

(
  cd "${DOWNLOAD_DIR}"
  sha256sum --check "${BASE_CHECKSUM}"
  printf '%s  %s\n' "${SUPPLEMENT_SHA256}" "${SUPPLEMENT_ARCHIVE}" | sha256sum --check -
)

tar --zstd -xf "${DOWNLOAD_DIR}/${BASE_ARCHIVE}" -C "${DEST}"
tar -xf "${DOWNLOAD_DIR}/${SUPPLEMENT_ARCHIVE}" -C "${DEST}"

BASE_ROOT="${DEST}/stage3_image_bundle"
SUPPLEMENT_ROOT="${DEST}/stage3_image_bundle_de_pathfix_20260805_v5_supplement"
mkdir -p "${BASE_ROOT}/images/de"
if ! cp -al "${SUPPLEMENT_ROOT}/images/de/." "${BASE_ROOT}/images/de/" 2>/dev/null; then
  cp -a "${SUPPLEMENT_ROOT}/images/de/." "${BASE_ROOT}/images/de/"
fi

printf 'Image root ready: %s\n' "${BASE_ROOT}"
printf 'Use: --image-root %q\n' "${BASE_ROOT}"

