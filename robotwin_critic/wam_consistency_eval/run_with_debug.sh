#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 COMMAND [ARG ...]" >&2
  echo "Runs an existing LingBot-VA evaluation command with WAM debug capture enabled." >&2
  exit 2
fi

export WAN_VA_SAVE_INFER_DEBUG=1
exec "$@"

