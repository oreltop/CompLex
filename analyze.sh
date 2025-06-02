#!/bin/bash

# Check if a repository path is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <repository_path>"
  exit 1
fi

# Navigate to the repository
repo_path="$1"
cd "$repo_path" || { echo "Invalid repository path: $repo_path"; exit 1; }

# Check if it's a git repository
if [ ! -d .git ]; then
  echo "Error: Not a git repository: $repo_path"
  exit 1
fi

# Print table header
printf "%-50s | %-15s | %-15s\n" "Filename" "Lines of Code" "Commits"
printf "%-50s-+-%-15s-+-%-15s\n" "$(printf '%50s' | tr ' ' '-')" "$(printf '%15s' | tr ' ' '-')" "$(printf '%15s' | tr ' ' '-')"

# Get list of all tracked files and process each one
git ls-files | while IFS= read -r file; do
  # Skip if file doesn't exist
  if [ ! -f "$file" ]; then
    continue
  fi
  
  # Try to detect if file is binary (simple method)
  if grep -q -m1 $'\x00' "$file" 2>/dev/null; then
    loc="binary"
  else
    # Count lines of code
    loc=$(wc -l < "$file")
  fi
  
  # Count commits for this file
  commits=$(git rev-list --count HEAD -- "$file")
  
  # Print file info
  printf "%-50s | %-15s | %-15s\n" "$file" "$loc" "$commits"
done
