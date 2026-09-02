#!/usr/bin/env bash
# Crop every PDF in the current working directory to its visible content.

set -euo pipefail

command -v pdfcrop >/dev/null 2>&1 || {
  echo "Error: pdfcrop is not installed." >&2
  exit 1
}

found=false
while IFS= read -r -d '' pdf; do
  found=true
  temporary=$(mktemp --tmpdir="$(pwd)" '.cropped.XXXXXX.pdf')
  if pdfcrop "$pdf" "$temporary" >/dev/null; then
    mv -- "$temporary" "$pdf"
    echo "Cropped: ${pdf#./}"
  else
    rm -f -- "$temporary"
    echo "Failed to crop: ${pdf#./}" >&2
    exit 1
  fi
done < <(find . -maxdepth 1 -type f -iname '*.pdf' -print0)

if [[ $found == false ]]; then
  echo "No PDF files found in: $(pwd)"
fi
