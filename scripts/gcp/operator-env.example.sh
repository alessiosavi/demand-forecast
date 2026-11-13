#!/bin/bash
# Copy outside the checkout, fill from project/catalogue evidence, then source it.
export GCP_PROJECT='replace-with-project-id'
export GCP_BUCKET='replace-with-single-region-bucket'
export GCP_ZONE='europe-west4-a'
export GCP_MACHINE='g2-standard-8'
export GCP_PROVISIONING='SPOT'
export GCP_IMAGE_PROJECT='deeplearning-platform-release'
export GCP_IMAGE_FAMILY='common-cu129-ubuntu-2204-nvidia-580'
export GCP_LEDGER="$HOME/.local/state/demand-forecast/gcp-ledger.json"
# Required USD values. Do not infer credit eligibility or prices from these names.
export GCP_CAP_USD='REPLACE'
export GCP_ENFORCE_USD='REPLACE'
export GCP_HOURLY_USD='REPLACE_WITH_ALL_IN_CATALOGUE_RATE'
export GCP_BOOT_DISK_TYPE='pd-balanced'
export GCP_BOOT_DISK_USD_PER_GB_MONTH='REPLACE'
export GCP_PRICE_SOURCE='REPLACE_WITH_CATALOGUE_SKU_IDS_AND_URL'
export GCP_PRICE_DATE='REPLACE_WITH_YYYY-MM-DD'
export GCP_JOB_URI="gs://$GCP_BUCKET/jobs/train.sh"
