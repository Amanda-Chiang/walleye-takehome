# Settings shared by the index, app and eval justfiles. Baseline config only.
#
# The recipes use `python` and `qasper-rag` from your active environment, so
# activate your virtual environment first. To use a smaller corpus or another
# index folder, set QASPER_CORPUS or QASPER_INDEX, e.g.
#   QASPER_CORPUS=results/small/corpus.jsonl QASPER_INDEX=.index-small just index build

python := env("PYTHON", "python")
config := "config/baseline.json"
corpus := env("QASPER_CORPUS", "data/corpus.jsonl")
index_dir := env("QASPER_INDEX", ".index")

# Build the baseline index. The public recipes depend on this one.
[private]
_build:
    qasper-rag index --corpus {{corpus}} --config {{config}} --index-dir {{index_dir}}
