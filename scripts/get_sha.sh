#!/usr/bin/env bash
set -euo pipefail
url=$URL
token=$TOKEN
request=$(curl -s -u \"$token\" $url)
restype=$(echo "$request" | jq .object.type | tr -d '"')
if [ $restype = tag ]; then
  url2=$(echo "$request" | jq .object.url | tr -d '"')
  request=$(curl -s -u \"$token\" $url2)
fi
sha=$(echo "$request" | jq .object.sha  | tr -d '"')
echo "sha=$sha"
