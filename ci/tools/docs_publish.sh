#!/usr/bin/env bash
# Publish one docs version folder to the gh-pages checkout in the current
# directory, or remove one.
#
# Usage: docs_publish.sh deploy VERSION BUILD_DIR
#        docs_publish.sh remove VERSION
#
# Set SITE_BASE to the URL path of the site root, such as /cubie/,
# GH_TOKEN to a token that can read pull requests, and GH_REPO to the
# repository.
#
# We may publish from several workflow runs at once, so we rebuild the
# commit on the newest gh-pages and push again whenever another run pushed
# first. We publish a pr-N folder only while pull request N is open, and
# check before every attempt, so that we never leave a preview online
# after its pull request closes.
set -euo pipefail

action="$1"
version="$2"
build_dir="${3:-}"
tools="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

for attempt in 1 2 3 4 5 6; do
  git fetch --quiet origin gh-pages
  git reset --quiet --hard origin/gh-pages
  git clean --quiet -fd

  rm -rf "$version"
  if [ "$action" = "deploy" ]; then
    if [[ "$version" == pr-* ]]; then
      state="$(gh pr view "${version#pr-}" --json state --jq .state)"
      if [ "$state" != "OPEN" ]; then
        echo "Pull request ${version#pr-} is $state; not publishing."
        exit 0
      fi
    fi
    cp -r "$build_dir" "$version"
  fi

  python "$tools/docs_site.py" index . "$SITE_BASE"
  git add -A
  if git diff --cached --quiet; then
    echo "Nothing to publish."
    exit 0
  fi
  git commit --quiet -m "$action $version: ${GITHUB_SHA:-local}"
  if git push --quiet origin HEAD:gh-pages; then
    exit 0
  fi
  echo "gh-pages moved on during attempt $attempt; retrying."
  sleep $((attempt * 5))
done

echo "Could not publish $version after 6 attempts." >&2
exit 1
