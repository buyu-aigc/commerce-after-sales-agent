"""Order lifecycle contract shared by policy checks and customer controls."""

PRE_SHIPMENT_STATUSES = frozenset({"created", "approved", "invoiced", "processing"})
