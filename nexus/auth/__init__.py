"""Authentication — per-user OAuth for the Assistant layer.

See docs/ASSISTANT-LAYER.md §6. Write actions run as the real user, never a
shared service account, so every token here is per-user and encrypted at rest.
"""
