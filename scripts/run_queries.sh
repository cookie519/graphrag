#!/bin/bash

# Path to the file containing queries
QUERY_FILE="/projects/JHA/shared/dataset/medmcqa_diabetes_llm_final/queries.txt"

# Initialize a counter
COUNTER=0

# Read each query from the file and execute the command
while IFS= read -r query; do
    # Increment the counter
    COUNTER=$((COUNTER + 1))

    echo "Running query $COUNTER: $query"
    graphrag query --root /scratch/gpfs/jx0800/data/graphrag --method local --query "$query"
    echo "--------------------------------------------------" # '-'*50

    # Break the loop after 100 queries
    if [ "$COUNTER" -eq 100 ]; then
        echo "Stopped after 100 queries."
        break
    fi
done < "$QUERY_FILE"