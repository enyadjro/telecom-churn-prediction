"""
Global configuration settings for the churn prediction pipeline.
"""

RANDOM_STATE = 42

# Business cost assumptions (used in threshold optimization)
COST_FALSE_NEGATIVE = 500   # missed churner (lost revenue)
COST_FALSE_POSITIVE = 50    # unnecessary retention offer
