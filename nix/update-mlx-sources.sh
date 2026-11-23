#!/usr/bin/env bash
# Refreshes every hash in nix/mlx-sources.json from its rev or URL. Run it after editing those.
set -euo pipefail
file="$(cd "$(dirname "$0")" && pwd)/mlx-sources.json"
updated=$(mktemp)
cp "$file" "$updated"
for name in $(jq -r 'keys[]' "$file"); do
  ref=$(jq -r --arg n "$name" \
    '.[$n] | if .github then "github:\(.github)/\(.rev)" else "tarball+\(.url)" end' "$file")
  hash=$(nix flake prefetch --json "$ref" | jq -r .hash)
  jq --arg n "$name" --arg h "$hash" '.[$n].hash = $h' "$updated" > "$updated.next"
  mv "$updated.next" "$updated"
  echo "$name: $ref"
done
mv "$updated" "$file"
