#!/usr/bin/env bash
# Delete the CI scratch directories that belong to a merged pull request.
#
# Every self-hosted lane names its scratch after the commit it was built from,
# or after the pull request number: salami's ~/ci-stage/<sha>-<variant>, the
# cross builder's cross-chipstar-work/{src,native,cross}-<sha>-<variant>, the
# library presubmit's /tmp/chipstar-ci-staging-salami/<pr>. Once the pull
# request is merged nothing will ever read those again, so the merge is the
# moment to delete them.
#
# Usage: reap-merged-dirs.sh [-n] -d DIR -p TEMPLATE [-p TEMPLATE ...] TOKEN...
#   -d DIR       directory holding the scratch trees
#   -p TEMPLATE  name to delete, with %s standing for the token; may glob.
#                Repeatable, so one call covers all of a lane's trees.
#   -n           dry run: report what would go, delete nothing
#
# TOKEN is a commit sha (hex, 7 to 40 characters) or a pull request number.
# Anything else is refused rather than expanded, so an empty or mangled
# workflow variable can never turn a template into a wildcard that matches
# every tree in DIR. A missing DIR and an empty token list are both no-ops.
#
# Example (the cross builder's trees for one merged commit):
#   reap-merged-dirs.sh -d /space/pvelesko/cross-chipstar-work \
#       -p 'src-%s-*' -p 'native-%s-*' -p 'cross-%s-*' 9ad35e7c5b2d
set -eu

DIR=""
DRY=0
TEMPLATES=()

usage() {
  [ -r "$0" ] && sed -n '2,/^set -eu/p' "$0" | sed 's/^# \{0,1\}//;$d' && return
  echo "usage: reap-merged-dirs.sh [-n] -d DIR -p TEMPLATE [-p TEMPLATE ...] TOKEN..."
}

while getopts 'd:p:nh' opt; do
  case "$opt" in
    d) DIR=$OPTARG ;;
    p) TEMPLATES+=("$OPTARG") ;;
    n) DRY=1 ;;
    h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))

[ -n "$DIR" ] || { echo "reap: -d DIR is required" >&2; exit 2; }
[ "${#TEMPLATES[@]}" -gt 0 ] || { echo "reap: at least one -p TEMPLATE is required" >&2; exit 2; }

if [ "$#" -eq 0 ]; then
  echo "reap: no commits or pull requests to reap, nothing to do"
  exit 0
fi

if [ ! -d "$DIR" ]; then
  echo "reap: $DIR does not exist, nothing to do"
  exit 0
fi

reaped=0
for token in "$@"; do
  # A pull request number is decimal and a sha is hex, so one rule covers both.
  case "$token" in
    ""|*[!0-9a-f]*)
      echo "reap: refusing token '$token': not a commit sha or PR number" >&2; exit 2 ;;
  esac
  [ "${#token}" -le 40 ] \
    || { echo "reap: refusing token '$token': too long for a sha" >&2; exit 2; }

  for tpl in "${TEMPLATES[@]}"; do
    # The token is hex or decimal by the check above, so it cannot introduce a
    # path separator or a glob character of its own.
    pat=${tpl//%s/$token}
    for p in "$DIR"/$pat; do
      [ -e "$p" ] || continue
      if [ "$DRY" = 1 ]; then
        echo "reap: would remove $p"
      else
        echo "reap: removing $p"
        rm -rf "$p"
      fi
      reaped=$((reaped + 1))
    done
  done
done

free=$(df -Pk "$DIR" 2>/dev/null | awk 'NR==2 {print $4}')
echo "reap: $DIR removed=$reaped for $# merged commit(s)/PR(s), ${free:-unknown} kB free"
