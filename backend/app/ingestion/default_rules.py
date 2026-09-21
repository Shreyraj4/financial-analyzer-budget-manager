"""Built-in merchant -> category rules. Kept free of database imports so training can use it at build time."""

RULES = [
    ("SWIGGY", "Food & Dining", "Delivery"),
    ("ZOMATO", "Food & Dining", "Delivery"),
    ("DMART", "Groceries", "Supermarket"),
    ("NETFLIX", "Subscriptions", "Streaming"),
    ("UBER", "Transport", "Rideshare"),
    ("AMAZON", "Shopping", "Online"),
    ("ELECTRICITY BOARD", "Utilities", "Electricity"),
    ("SALARY", "Income", None),
]
