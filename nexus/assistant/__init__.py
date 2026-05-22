"""The Nexus Assistant — conversational + action layer.

See docs/ASSISTANT-LAYER.md for the full design. This package is the
"Assistant layer" and "Capability layer" of that design:

  models.py          ActionProposal / Conversation / message models
  store.py           SQLite persistence for conversations + proposals
  connector_port.py  the boundary to external systems (Jira/Confluence)
  capabilities.py    the curated, intent-shaped tool facade (Tier-2)
  loop.py            the tool-calling agent loop (the "brain")
"""
