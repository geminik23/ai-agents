#!/usr/bin/env bash
# Prepare a disposable CI clock. The test binary still evaluates its own kernel evidence.
set -euo pipefail
[[ ${RUNNER_OS:-} == macOS && ${CLOCK_RUNNER_ENVIRONMENT:-} == github-hosted ]] || {
  echo "This clock setup is limited to a GitHub-hosted Mac." >&2
  exit 1
}

export HOMEBREW_NO_AUTO_UPDATE=1
brew install ntp
ntp_prefix=$(brew --prefix ntp)
export PATH="$ntp_prefix/sbin:$ntp_prefix/bin:$PATH"
clock_dir=$(mktemp -d "${RUNNER_TEMP:?}/restore-clock.XXXXXX")

# ntpd may correct wall time. Use UTC for correlation and a monotonic clock for elapsed time and deadlines.
monotonic_seconds() { python3 -c 'import time; print(time.monotonic_ns() // 10**9)'; }
started=$(monotonic_seconds)
log_event() {
  printf '%s elapsed=%ss %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$(( $(monotonic_seconds) - started ))" "$*"
}

kernel_reports_time_ok() {
  ntptime > "$clock_dir/kernel.log" 2>&1 &&
    grep -Fq 'ntp_adjtime() returns code 0 (OK)' "$clock_dir/kernel.log"
}

if kernel_reports_time_ok; then
  log_event "first kernel TIME_OK; already synchronized, starting strict tests"
  cat "$clock_dir/kernel.log"
  exec "$@"
fi
cat "$clock_dir/kernel.log"
setup=/usr/sbin/systemsetup
server_setting=$(sudo -n "$setup" -getnetworktimeserver)
server=${server_setting#Network Time Server: }
[[ $server != "$server_setting" && $server =~ ^[A-Za-z0-9][A-Za-z0-9.:-]*$ ]] || {
  echo "No usable OS-configured NTP server: $server_setting" >&2
  exit 1
}
comparison_server=time.windows.com
printf 'NTP sources: OS=%s comparison=%s\n' "$server" "$comparison_server"
previous=$(sudo -n "$setup" -getusingnetworktime)
case "$previous" in
  'Network Time: On'|'Network Time: Off') ;;
  *) echo "Cannot determine the previous network time setting: $previous" >&2; exit 1 ;;
esac

clock_pid=
cleanup() {
  if [[ -n $clock_pid ]]; then
    sudo -n kill -TERM "$clock_pid" 2>/dev/null || true
    wait "$clock_pid" || true
  fi
  if [[ $previous == 'Network Time: On' ]]; then sudo -n "$setup" -setusingnetworktime on; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ $previous == 'Network Time: On' ]]; then sudo -n "$setup" -setusingnetworktime off; fi

# Keep the steady polling interval bounded at the 64 seconds observed in CI #3.
poll_exponent=6
printf 'server %s iburst maxpoll %s\n' "$server" "$poll_exponent" > "$clock_dir/ntp.conf"
if [[ $(printf '%s' "${server%.}" | tr '[:upper:]' '[:lower:]') != "$comparison_server" ]]; then
  printf 'server %s iburst maxpoll %s\n' "$comparison_server" "$poll_exponent" >> "$clock_dir/ntp.conf"
else
  echo "OS and comparison source are the same hostname; using one association."
fi
cat >> "$clock_dir/ntp.conf" <<EOF
logconfig +sysevents +peerevents
restrict default ignore
restrict -6 default ignore
restrict source nomodify noquery notrap
restrict 127.0.0.1
restrict ::1
EOF
# Link the selected numeric peer to its configured hostname and its own send/receive counters.
log_associations() {
  local associations id
  associations=$(ntpq -n -c associations) || return 0
  printf '%s\n' "$associations"
  while read -r id; do
    ntpq -n -c "rv $id srchost,srcadr,reach,stratum" -c "pstats $id" || true
  done < <(awk '$1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ {print $2}' <<< "$associations")
}

# Acquisition is bounded independently of training. Start the training allowance only once,
# at the first observed system peer, the observable proxy for the first discipline update.
acquisition_seconds=300
training_seconds=300  # ntpd 4.2.8p18 default stepout
poll_seconds=$((1 << poll_exponent))
deadline=$(( $(monotonic_seconds) + acquisition_seconds ))
first_selection_seen=false
kernel_ok_seen=false
responders=" "
previous_peer=
failure="no system peer selected within the acquisition window"
log_event "starting ntpd; peer acquisition budget=${acquisition_seconds}s"
sudo -n "$(command -v ntpd)" -n -g -c "$clock_dir/ntp.conf" > "$clock_dir/ntpd.log" 2>&1 &
clock_pid=$!
while (( $(monotonic_seconds) < deadline )); do
  if ! sudo -n kill -0 "$clock_pid" 2>/dev/null; then
    failure="ntpd exited during clock preparation"
    break
  fi
  peer=
  if ntpq -pn > "$clock_dir/peers.log"; then
    # A nonzero octal reach register records a usable reply; raw receive counters alone do not.
    while read -r address reach selected; do
      if [[ $responders != *" $address "* ]]; then
        log_event "first valid response observed peer=$address reach=$reach"
        responders+="$address "
      fi
      if [[ $selected == yes ]]; then peer=$address; fi
    done < <(awk 'NF == 10 && $4 == "u" && $7 ~ /^[0-7]+$/ && $7+0 > 0 {
      address=$1; sub(/^[*+x#.~o-]/, "", address)
      print address, $7, ($1 ~ /^\*/ ? "yes" : "no")
    }' "$clock_dir/peers.log")
  fi
  observed_at=$(monotonic_seconds)
  if (( observed_at >= deadline )); then break; fi
  if [[ $peer != "$previous_peer" ]]; then
    log_event "system peer=${peer:-none}"
    previous_peer=$peer
  fi
  if [[ -n $peer && $first_selection_seen == false ]]; then
    first_selection_seen=true
    # One update can finish frequency training and another can publish synchronized kernel state.
    deadline=$((observed_at + training_seconds + 2 * poll_seconds))
    failure="kernel TIME_OK and a current system peer did not coincide within the training window"
    log_event "training budget=${training_seconds}s plus two ${poll_seconds}s polls"
  fi
  if kernel_reports_time_ok; then
    if [[ $kernel_ok_seen == false ]]; then
      log_event "first kernel TIME_OK observed"
      kernel_ok_seen=true
    fi
    if [[ -n $peer ]] && (( $(monotonic_seconds) < deadline )); then
      cat "$clock_dir/peers.log" "$clock_dir/kernel.log"
      log_associations
      cat "$clock_dir/ntpd.log" || true
      log_event "kernel TIME_OK with peer=$peer; starting strict tests"
      "$@"
      exit 0
    fi
  fi
  sleep 2
done
log_event "clock preparation stopped"
echo "::error::NTP preparation failed: $failure"
ntpq -pn || true
log_associations
ntptime || true
cat "$clock_dir/ntpd.log" || true
exit 1
