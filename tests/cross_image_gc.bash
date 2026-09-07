#!/bin/bash
# The Mali workflow trims cross-build images, each about 1.6 GB, keeping the
# newest few. It sorted on {{.CreatedAt}}, which is the IMAGE's timestamp and
# not the moment a tag was written, so a rebuild whose layers all hit the cache
# produced the same image as before and the tag just created inherited an old
# date. It then sorted into the delete list and was removed between the build
# step and the step that runs it, which failed with "Unable to find image
# chipstar-cross-aarch64:<tag> locally".
#
# Drives the workflow's own pipeline, extracted from the YAML rather than
# copied, against a fixture image list. Needs no docker and no network.
#
# Usage: cross_image_gc.bash <unit-tests-arm64-mali.yml>
set -u
WF="${1:?path to the Mali workflow}"

[ -f "${WF}" ] || { echo "FAIL: no such workflow: ${WF}"; exit 1; }
WF=$(cd "$(dirname "${WF}")" && pwd)/$(basename "${WF}")
# The fixture goes in a directory this script creates and removes, not one a
# caller names: emptying a directory somebody else owns is not this test's to
# do, and the fixture has to start empty to be repeatable.
OUT=$(mktemp -d) || { echo "FAIL: cannot create a work directory"; exit 1; }
trap 'rm -rf "${OUT}"' EXIT
cd "${OUT}" || exit 1

# The filter chain between `docker images` and `xargs docker rmi`: everything
# that decides WHICH tags die. Taken from the workflow so a change there is
# either reflected here or breaks this test.
sed -n "/docker images chipstar-cross-aarch64 --format/,/xargs -r -I{} docker rmi/p" "${WF}" \
  | sed -e 's/^ *//' -e 's/\\$//' > chain.txt
if ! grep -q "grep -v" chain.txt; then
  echo "FAIL: could not extract the image GC pipeline from ${WF}"
  exit 1
fi
# Drop the docker calls at either end; feed the fixture in and read tags out.
FILTER=$(sed -e '/docker images/d' -e '/xargs/d' chain.txt | tr -d '\n')
if [ -z "${FILTER}" ]; then echo "FAIL: extracted an empty GC filter"; exit 1; fi

# Two tags older than the one being built, which is what a cache-hit rebuild
# looks like: same image, so the new tag carries the old creation date.
# newtag carries an OLD date on purpose: that is what a cache-hit rebuild
# produces, and it is what sorted the tag into the delete list. Enough other
# tags to leave something for the GC to trim once newtag and base are excluded.
cat > images.txt <<'IMAGES'
newtag 2026-08-19 10:30:25 +0300 EEST
oldtag-a 2026-09-04 12:42:46 +0300 EEST
oldtag-b 2026-09-01 09:00:00 +0300 EEST
oldtag-c 2026-08-19 10:30:25 +0300 EEST
base 2026-08-19 09:14:48 +0300 EEST
IMAGES

TAG=newtag
export TAG
DOOMED=$(eval "cat images.txt ${FILTER}")
echo "tags the GC would delete: [$(echo ${DOOMED} | tr '\n' ' ')]"

if echo "${DOOMED}" | grep -qx "${TAG}"; then
  echo "FAIL: the GC deletes ${TAG}, the image this run just built and is about to use"
  exit 1
fi
if ! echo "${DOOMED}" | grep -qx "oldtag-c"; then
  echo "FAIL: the GC no longer trims anything; it must still remove older images"
  exit 1
fi
if echo "${DOOMED}" | grep -qx "base"; then
  echo "FAIL: the GC must never delete the base image"
  exit 1
fi
echo "PASSED"
