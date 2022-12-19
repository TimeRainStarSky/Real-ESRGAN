cd "$(dirname "$0")"
exec poetry run python inference_realesrgan.py "$@"