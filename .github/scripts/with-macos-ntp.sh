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

kernel_reports_time_ok() {
  ntptime > "$clock_dir/kernel.log" &&
    grep -Fq 'ntp_adjtime() returns code 0 (OK)' "$clock_dir/kernel.log"
}

if kernel_reports_time_ok; then
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

cat > "$clock_dir/ntp.conf" <<EOF
server $server iburst
restrict default ignore
restrict -6 default ignore
restrict source nomodify noquery notrap
restrict 127.0.0.1
restrict ::1
EOF
# Continuous ntpd maintains kernel discipline while the producer and consumer run.
sudo -n "$(command -v ntpd)" -n -g -c "$clock_dir/ntp.conf" > "$clock_dir/ntpd.log" 2>&1 &
clock_pid=$!
deadline=$((SECONDS + 300))
while (( SECONDS < deadline )); do
  if ! sudo -n kill -0 "$clock_pid" 2>/dev/null; then break; fi
  if ntpq -pn > "$clock_dir/peers.log" && grep -q '^\*' "$clock_dir/peers.log" && kernel_reports_time_ok; then
    cat "$clock_dir/peers.log" "$clock_dir/kernel.log"
    "$@"
    exit 0
  fi
  sleep 2
done
echo "::error::NTP preparation did not establish a selected peer and synchronized kernel clock."
ntpq -pn || true
ntptime || true
tail -40 "$clock_dir/ntpd.log" || true
exit 1
