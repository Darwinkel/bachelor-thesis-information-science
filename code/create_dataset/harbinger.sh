#!/bin/bash
for file in domains/tranco_*
do
  python3 collector.py "$file"
done