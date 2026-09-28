"""Interface limits and defaults. Change them here, not inside the pages."""

APP_TITLE = "Developer Documentation Assistant"
APP_TAGLINE = (
    "Ask questions and receive answers grounded in your indexed documentation.")

# Questions longer than this are rejected before any API call.
MAX_QUESTION_CHARS = 1000

# The file types rag.loader.load_document can actually read.
ALLOWED_EXTENSIONS: tuple[str, ...] = (".pdf", ".txt", ".md")

# Keep in sync with server.maxUploadSize in .streamlit/config.toml, which
# makes Streamlit refuse larger files before they reach the app.
MAX_UPLOAD_MB = 10
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_FILES_PER_BATCH = 10

# Chunks sent to the embedding API per request.
EMBEDDING_BATCH_SIZE = 100

# Retrieval settings shown in the sidebar. The default matches the app before
# Stage 11. These settings never change the Stage 10 evaluation configuration.
DEFAULT_TOP_K = 3
MIN_TOP_K = 1
MAX_TOP_K = 10

# Display limits.
MAX_EXCERPT_CHARS = 400
FEEDBACK_PAGE_SIZES: tuple[int, ...] = (10, 25, 50)
QUESTION_PREVIEW_CHARS = 90
MAX_FAILURE_ROWS = 50
MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024

# Reading every chunk's metadata to count documents is only done for
# collections up to this size; above it the document count is not shown.
MAX_STATUS_SCAN = 10_000
