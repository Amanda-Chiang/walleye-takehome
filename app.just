import 'common.just'

# Start the chat page. The index is built first if it is missing or out of date.
start port="8000": _build
    qasper-rag serve --index-dir {{index_dir}} --config {{config}} --port {{port}}
