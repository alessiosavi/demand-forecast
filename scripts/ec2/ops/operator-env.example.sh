#!/bin/bash
set -u
# Copy outside the repository, fill the inspected values, then source it or export
# OPERATOR_ENV=/absolute/path/to/operator-env.sh. No provisioning is done here.
export EXPERIMENT=/absolute/path/to/experiment
export LOCAL_CODE=/absolute/path/to/demand-forecast
export PYTHON=python3
export AWS_PROFILE=your-profile AWS_REGION=eu-central-1
export AWS_DEFAULT_OUTPUT=json AWS_PAGER=''
export OPERATOR_IP=your-public-ip VPC_ID=your-vpc AMI_ID=your-ami
export SUBNET_ID=''                         # Optional, subject to runbook §3.
export LEDGER="$EXPERIMENT/ops/cost-ledger.json"
guard() { "$PYTHON" "$LOCAL_CODE/scripts/ec2/cost_guard.py" --ledger "$LEDGER" "$@"; }
launch() { bash "$LOCAL_CODE/scripts/ec2/pretrain_launch.sh" "$@" --ledger "$LEDGER"; }

# Existing guest layout from runbook §4; env.sh and Colab inputs must exist.
export GUEST_USER=ubuntu GUEST_HOME=/home/ubuntu
export GUEST_CODE="$GUEST_HOME/fev/code" GUEST_ENV="$GUEST_HOME/fev/env.sh"
export GUEST_DISPATCH="$GUEST_HOME/dispatch-job.sh" GUEST_WAIT="$GUEST_HOME/wait-job.sh"
export GUEST_LOG_ROOT="$GUEST_HOME/fev/host-logs"
export REPO_URL=https://github.com/alessiosavi/demand-forecast.git
export SETUP_TIMEOUT_SECONDS=2400 COPY_MARGIN_SECONDS=120
export CHECKOUT_TIMEOUT_SECONDS=180 PUSH_TIMEOUT_SECONDS=180
export DISPATCH_TIMEOUT_SECONDS=60 COPY_TIMEOUT_SECONDS=180 COPY_BUDGET_SECONDS=240
export SSH_TIMEOUT_SECONDS=120 GUARD_TIMEOUT_SECONDS=180 STOP_TIMEOUT_SECONDS=480
# SSH stop fallback brings the guest deadline forward through arm-deadline.sh;
# allow up to about two minutes. DEADLINE_SET means scheduled, not a confirmed EC2 stop.
export START_TIMEOUT_SECONDS=1200 START_RETRIES=24
# Optional: latest reservation deadline in UTC; retries shorten the session.
# export LATEST_DEADLINE=2026-11-12T02:00:00Z

# Inputs to run-guest-job.sh. JSON arrays preserve spaces without shell eval.
export LABEL=backtests MINUTES=60 SHA='<full approved 40-character SHA>'
export GUEST_SCRIPT="$LOCAL_CODE/scripts/ec2/ops/guest/wins-guest.sh"
export GUEST_ARGS='["backtests", "/home/ubuntu/wins/manifest.json"]'
export PUSH='[{"files":["/absolute/path/to/manifest.json"],"remote_dir":"/home/ubuntu/wins"}]'
export PULL='[{"remote_dir":"/home/ubuntu/fev/runs/wins","local_dir":"/absolute/path/to/experiment/rows","exclude":["*.npz"]}]'
export EXPECTED_EXIT=0 SKIP_SETUP=0
# wins-guest.sh uses $OPS/wins/<stage> for logs/status; optional WINS_STATE_DIR
# overrides $OPS/wins. Manifest cmd is an argv array or a shell-quoted argv string.
# On the guest, WINS_LANES sets concurrent light lanes (1–8, default 3); heavy
# items still run alone. Failed items retry once alone, in manifest order, after
# all first attempts; launch errors (exit 127 from OSError) are not retried.
# Set WINS_RETRY_FAILED=0 on the guest to disable retries. Logs keep both attempts;
# markers, WINS_EXIT, WINS_DONE and queue exit reflect the final attempt/state.
# fev run/tune receive --resume automatically. A non-zero queue exit means at least
# one item still failed; results are still copied back.
# Each invocation lets fev validate and resume its own output artifacts.
