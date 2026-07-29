# Gunicorn configuration — auto-loaded when gunicorn is started in this directory.
# No command-line flags needed; Render's "gunicorn main:app" will pick this up automatically.

# Worker timeout in seconds.
# The voice pipeline runs two sequential Gemini API calls (STT + LLM) plus edge-tts,
# which can take 30-60 seconds on a cold Render free tier instance.
timeout = 120

# Number of worker processes (1 is safe for Render free tier with limited RAM)
workers = 1
