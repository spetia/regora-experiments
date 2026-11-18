#!/bin/bash
# The complete unattended queue: stages D, G, E, F and H, then publish.sh. Every stage resumes where it stopped and
# skips what is done, so this is also the one command to run after a restart:
#
#   bash queue_all.sh "<path to a worktree of the experiments repository on the scaling-curve-study branch>"
#
# Before each step it waits until no study process has run for two checks a minute apart, so two stages never
# train at the same time.
cd "$(dirname "$0")" || exit 1
WT="$1"
PY=.venv/Scripts/python

count() {  # running processes of the given image whose command line matches the regex
  powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name like '$1%'\" | Where-Object { \$_.CommandLine -match '$2' } | Measure-Object).Count" 2>/dev/null | tr -d '\r '
}

wait_idle() {
  local idle=0
  while [ "$idle" -lt 2 ]; do
    if [ "$(count python 'stage_[defgh]\.py|run_all\.py|study\.py')" = "0" ]; then idle=$((idle + 1)); else idle=0; fi
    sleep 60
  done
}

step() {
  wait_idle
  echo "$(date '+%F %T') start: $*"
  "$@"
  echo "$(date '+%F %T') exit $?: $*"
}

step $PY -u stage_d.py all >> run_D.log 2>&1
step $PY -u stage_g.py run >> run_G.log 2>&1  # before E and F: the most decision-relevant open question
step $PY -u stage_e.py run >> run_E.log 2>&1
step $PY -u stage_f.py run >> run_F.log 2>&1
step $PY -u stage_h.py run >> run_H.log 2>&1
wait_idle
bash ./publish.sh "$WT"
echo "$(date '+%F %T') QUEUE DONE"
