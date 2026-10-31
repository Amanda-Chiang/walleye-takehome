import 'common.just'

# Score the baseline on the first <count> dev questions. Builds the index first.
run count="10": _build
    {{python}} tools/run_eval.py --existing-index --index-dir {{index_dir}} --config {{config}} --count {{count}}

# Same as run, and also score with RAGAS (needs: pip install --editable ".[ragas]").
ragas count="10": _build
    {{python}} tools/run_eval.py --existing-index --index-dir {{index_dir}} --config {{config}} --count {{count}} --ragas
