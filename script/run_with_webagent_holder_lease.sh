#!/usr/bin/env bash
set -euo pipefail

CONTROL_FILE="${CONTROL_FILE:?Set CONTROL_FILE to the Deployment pause_until file}"
LEASE_SECONDS="${LEASE_SECONDS:-120}"
REFRESH_SECONDS="${REFRESH_SECONDS:-30}"
SETTLE_SECONDS="${SETTLE_SECONDS:-6}"

if (( LEASE_SECONDS <= REFRESH_SECONDS )); then
  echo "LEASE_SECONDS must exceed REFRESH_SECONDS" >&2
  exit 2
fi
if (( $# == 0 )); then
  echo "Usage: CONTROL_FILE=... $0 command [args...]" >&2
  exit 2
fi

CONTROL_DIR="$(dirname "$CONTROL_FILE")"
SMART_MANUAL_FILE="$CONTROL_DIR/manual_pause_until"
SMART_CLEAR_FILE="$CONTROL_DIR/clear_auto"
INLINE_DISABLED_DEADLINE="${INLINE_DISABLED_DEADLINE:-4102444800}"
SMART_HOLDER_ACTIVE=0
mkdir -p "$CONTROL_DIR"
if pgrep -f "python.*${CONTROL_DIR}/smart_holder[.]py" >/dev/null 2>&1; then
  SMART_HOLDER_ACTIVE=1
fi
lease_pid=""
child_pid=""

write_atomic() {
  local path="$1"
  local value="$2"
  local temporary
  temporary="${path}.lease.$$"
  printf '%s\n' "$value" > "$temporary"
  mv "$temporary" "$path"
}

write_deadline() {
  local deadline
  deadline="$(( $(date +%s) + LEASE_SECONDS ))"
  if [[ "$SMART_HOLDER_ACTIVE" == "1" ]]; then
    write_atomic "$CONTROL_FILE" "$INLINE_DISABLED_DEADLINE"
    write_atomic "$SMART_MANUAL_FILE" "$deadline"
  else
    write_atomic "$CONTROL_FILE" "$deadline"
  fi
}

lease_loop() {
  while true; do
    write_deadline
    sleep "$REFRESH_SECONDS"
  done
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [[ -n "$lease_pid" ]]; then
    kill "$lease_pid" 2>/dev/null || true
    wait "$lease_pid" 2>/dev/null || true
  fi
  if [[ -n "$child_pid" ]]; then
    # The session leader may exit before one of its workers. Always signal its
    # process group so failed launches cannot leave orphaned GPU workers behind.
    kill -TERM -- "-$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  if [[ "$SMART_HOLDER_ACTIVE" == "1" ]]; then
    write_atomic "$CONTROL_FILE" "$INLINE_DISABLED_DEADLINE"
    write_atomic "$SMART_MANUAL_FILE" 0
    # Clear the smart holder's automatic cooldown only after the workload
    # process group has been drained. Otherwise a surviving worker can
    # immediately recreate the cooldown and leave idle GPUs unprotected.
    : > "$SMART_CLEAR_FILE"
  else
    write_atomic "$CONTROL_FILE" 0
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

echo "HOLDER_LEASE_MODE smart_holder=$SMART_HOLDER_ACTIVE control=$CONTROL_FILE"
write_deadline
lease_loop &
lease_pid=$!
sleep "$SETTLE_SECONDS"

# Keep the lease wrapper outside the workload process group. This lets TERM on
# the wrapper cleanly stop the complete workload tree before releasing holder.
setsid "$@" &
child_pid=$!
wait "$child_pid"
child_pid=""
