#!/bin/bash

DATABASE_FILE="training_data.csv"
MEMORY_DUMP_FILE_PREFIX="memory_dump_"
TEST_FILE_PREFIX="test_file_"

echo "Starting cleanup..."

if [ -f "$DATABASE_FILE" ]; then
  echo "Removing database file: $DATABASE_FILE"
  rm "$DATABASE_FILE"
else
  echo "Database file not found: $DATABASE_FILE"
fi

find . -maxdepth 1 -type f -name "$MEMORY_DUMP_FILE_PREFIX*.bin" -print0 | while IFS= read -r -d $'\0' file; do
    echo "Removing memory dump file: $file"
    rm "$file"
done

find . -maxdepth 1 -type f -name "$TEST_FILE_PREFIX*" -print0 | while IFS= read -r -d $'\0' file; do
  echo "Removing test file: $file"
  rm "$file"
  if [ -f "$file.enc" ]; then
    echo "Removing encrypted test file: $file.enc"
    rm "$file.enc"
  fi
done

echo "Cleanup complete."
