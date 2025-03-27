# instructions and descriptions of all scripts

## launch grapgrag
ollama serve
cp -r /projects/JHA/shared/graph/pubmed/injected_expanded/* /scratch/gpfs/jx0800/data/graphrag/output/
scripts/run_queries.sh > /scratch/gpfs/jx0800/data/graphrag/results/injected_expanded_query_output_8B.log 2>&1

## local search
Graphrag.query.structured_search.local_search.mixed_context.py selected_entities = map_query_to_entities()
extracted entity number = oversample_scaler * k
