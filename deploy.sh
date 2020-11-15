#!/bin/sh

# If a command fails then the deploy stops
set -e

printf "\033[0;32mDeploying updates to GitHub...\033[0m\n"

# Add changes to git.
git add .

# Commit changes.
# msg="rebuilding site $(date)"
msg="Release $(date '+%Y/%m/%d %T')"
if [ -n "$*" ]; then
	msg="$*"
fi
git commit -m "$msg"

# Push
git push origin master

# tag
tag_name="v1.0.6"
git tag $tag_name
git push origin $tag_name
