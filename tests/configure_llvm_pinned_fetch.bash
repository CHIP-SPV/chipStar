#!/bin/bash
# configure_llvm.sh reuses an existing llvm-project checkout rather than
# recloning it, but it fetched unconditionally before checking the pinned ref
# out. For a pinned TAG that is already local that fetch retrieves nothing: it
# only adds a network round trip, and it fails the whole build when GitHub is
# unreachable or throttling the runner. For a moving BRANCH the fetch is
# required, or the build would silently keep compiling stale source.
#
# Drives the real fetch_pinned_ref, extracted from configure_llvm.sh itself
# rather than copied here, against fixture repositories whose remote does not
# resolve. A skipped fetch therefore succeeds and an attempted one is recorded.
#
# Needs git and nothing else: no GPU, no toolchain, no network.
#
# Usage: configure_llvm_pinned_fetch.bash <configure_llvm.sh> <work-dir>
set -u
SCRIPT="${1:?path to configure_llvm.sh}"
OUT="${2:?work directory}"

[ -f "${SCRIPT}" ] || { echo "FAIL: no such script: ${SCRIPT}"; exit 1; }
mkdir -p "${OUT}" || { echo "FAIL: cannot create ${OUT}"; exit 1; }
# Absolute before the cd below, or a relative argument would resolve against
# the work directory instead of the caller's.
SCRIPT=$(cd "$(dirname "${SCRIPT}")" && pwd)/$(basename "${SCRIPT}")
OUT=$(cd "${OUT}" && pwd)
rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}" || exit 1

# The function's real text. Renaming or removing it fails here rather than
# leaving this test quietly exercising nothing.
sed -n '/^fetch_pinned_ref() {/,/^}/p' "${SCRIPT}" > fn.sh
if [ ! -s fn.sh ]; then
  echo "FAIL: fetch_pinned_ref not found in ${SCRIPT}"
  exit 1
fi

# Stand-in for retry. It records the exact command it was handed and does NOT
# run it, so the test never touches the network and can assert what would have
# been executed rather than merely that something was.
{ echo 'retry() { echo "$*" >> "${ATTEMPTS}"; return "${RETRY_RC:-0}"; }'
  cat fn.sh; } > fn_under_test.sh

git init -q upstream || { echo "FAIL: fixture init"; exit 1; }
(
  cd upstream
  git -c user.email=t@t -c user.name=t commit -q --allow-empty -m one
  git tag pinned-tag
  git branch moving-branch
) || { echo "FAIL: fixture setup"; exit 1; }
git clone -q upstream clone 2>/dev/null || { echo "FAIL: fixture clone"; exit 1; }
git -C clone remote set-url origin https://no-such-host.invalid/nope.git

# Echoes: <exit status> <number of fetches attempted> <command of the first>
run_case() {
  export ATTEMPTS="${OUT}/attempts.$1"
  : > "${ATTEMPTS}"
  ( cd clone && . "${OUT}/fn_under_test.sh" && fetch_pinned_ref "$1" >/dev/null 2>&1 )
  local rc=$?
  local calls cmd
  calls=$(grep -c . "${ATTEMPTS}" 2>/dev/null)
  cmd=$(head -1 "${ATTEMPTS}" 2>/dev/null)
  echo "$rc ${calls:-0} ${cmd:-none}"
}

FAILED=0

# A textual check, and only that: it reads the script rather than running it.
# It catches the shape of the regression this test exists for, a reuse-path
# caller fetching directly, and it catches a stray fetch appearing anywhere
# else in the file. It cannot prove the callers behave correctly, because
# driving them means running configure_llvm.sh's reuse path, which needs a
# populated llvm-project and a patch series that applies, and that is out of
# reach of a test meant to run before every LLVM build. Treat a pass here as
# "nothing obviously reaches around the helper", not as caller coverage.
# Each reuse-path `cd` into a repository must be followed by the helper. The
# clone path also cds into llvm-project and correctly does not fetch, so only
# lines whose NEXT line touches the network are of interest: the check is that
# no `cd` into a repo is followed by a bare fetch.
while IFS= read -r next; do
  case "${next}" in
    *"git fetch"*)
      echo "FAIL: a reuse-path caller fetches directly instead of via fetch_pinned_ref:"
      echo "      ${next}"
      FAILED=1 ;;
  esac
done < <(grep -A1 -E '^[[:space:]]+cd (llvm-project|llvm/projects/SPIRV-LLVM-Translator)$' "${SCRIPT}" \
         | grep -vE '^[[:space:]]*cd |^--$')
# -E with a single alternation group: `\|` is a GNU extension that BSD grep,
# and so macOS, does not honour, and there it silently matches nothing.
# Trailing whitespace or a trailing comment must not hide a fetch from this.
FETCHES=$(grep -cE '^[[:space:]]*(retry )?git fetch origin([[:space:]]|#|$)' "${SCRIPT}")
if [ "${FETCHES}" -ne 1 ]; then
  echo "FAIL: expected exactly one 'git fetch origin' in the script, found ${FETCHES};"
  echo "      a caller is fetching without going through fetch_pinned_ref"
  FAILED=1
fi

read -r RC CALLS CMD <<< "$(run_case pinned-tag)"
echo "pinned tag already local -> rc=${RC} fetches=${CALLS} cmd=[${CMD}]"
if [ "${RC}" -ne 0 ] || [ "${CALLS}" -ne 0 ]; then
  echo "FAIL: a tag that is already local must not be fetched"; FAILED=1
fi

read -r RC CALLS CMD <<< "$(run_case moving-branch)"
echo "moving branch            -> rc=${RC} fetches=${CALLS} cmd=[${CMD}]"
if [ "${CALLS}" -ne 1 ] || [ "${CMD}" != "git fetch origin" ]; then
  echo "FAIL: a branch must be fetched with 'git fetch origin', got ${CALLS} x [${CMD}]"; FAILED=1
fi

read -r RC CALLS CMD <<< "$(run_case absent-ref)"
echo "ref not present locally  -> rc=${RC} fetches=${CALLS} cmd=[${CMD}]"
if [ "${CALLS}" -ne 1 ] || [ "${CMD}" != "git fetch origin" ]; then
  echo "FAIL: an absent ref must be fetched with 'git fetch origin'"; FAILED=1
fi

# A name that is both a tag and a branch here would be resolved ambiguously by
# the checkout that follows, so it must fetch rather than trust the tag.
git -C clone branch -q ambiguous 2>/dev/null
git -C clone tag ambiguous 2>/dev/null
read -r RC CALLS CMD <<< "$(run_case ambiguous)"
echo "same name tag AND branch -> rc=${RC} fetches=${CALLS} cmd=[${CMD}]"
if [ "${CALLS}" -ne 1 ]; then
  echo "FAIL: an ambiguous ref must be fetched, not assumed to be the tag"; FAILED=1
fi

# A fetch that keeps failing must fail the build rather than proceed onto stale
# source: the script runs under set -e and the caller relies on that.
export RETRY_RC=1
read -r RC CALLS CMD <<< "$(run_case moving-branch)"
unset RETRY_RC
echo "fetch fails              -> rc=${RC} fetches=${CALLS}"
if [ "${RC}" -eq 0 ]; then
  echo "FAIL: a failed fetch must propagate, not be swallowed"; FAILED=1
fi

if [ "${FAILED}" -eq 0 ]; then echo "PASSED"; exit 0; fi
exit 1
