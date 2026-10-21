#!/usr/bin/env python3
"""Estimate using a user-supplied current price; never queries/provisions AWS."""

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--instance", choices=["p5.4xlarge", "p5.48xlarge"], required=True)
parser.add_argument("--price-per-hour", type=float, required=True)
parser.add_argument("--hours", type=float, required=True)
args = parser.parse_args()
if args.price_per_hour <= 0 or args.hours <= 0:
    parser.error("Price and duration must be positive")
print(
    json.dumps(
        {
            "instance": args.instance,
            "hours": args.hours,
            "user_supplied_price_per_hour": args.price_per_hour,
            "estimated_instance_cost": args.hours * args.price_per_hour,
            "excludes": "storage, network, tax, retries, queue; no quota/credit assumptions",
        },
        indent=2,
    )
)
