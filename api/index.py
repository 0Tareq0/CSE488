from fastapi import FastAPI
from backend.app.main import app as _app

# Vercel needs a standard way to recognize FastAPI app.
# The `app` variable here will be picked up by @vercel/python
app = _app
