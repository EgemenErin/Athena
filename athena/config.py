MODEL = "qwen2.5-coder:7b"
MAX_RETRIES = 1
MAX_REVIEW_RETRIES = 1
ENABLE_AGENT_PIPELINE = True

# Cleaning: batched per-column AI (no global action cap)
CLEANING_BATCH_SIZE = 30
CLEANING_NUM_PREDICT_PER_BATCH = 4096
