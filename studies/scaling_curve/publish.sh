#!/bin/bash
# Write the analysis of every finished stage to results/, commit, and (given a worktree of the experiments
# repository on the scaling-curve-study branch) copy the study there, commit and push.
#
#   ./publish.sh [path/to/worktree]
#
# Used to finish an unattended run: nobody has read the outputs it pushes, and the commit message says so.
cd "$(dirname "$0")" || exit 1
PY=.venv/Scripts/python
$PY analyze.py > results/analysis_ABC.txt 2>&1
$PY analyze_d.py > results/analysis_D.txt 2>&1
$PY stage_e.py analyze > results/analysis_E.txt 2>&1
$PY stage_f.py analyze > results/analysis_F.txt 2>&1
$PY stage_g.py analyze > results/analysis_G.txt 2>&1
$PY stage_h.py analyze > results/analysis_H.txt 2>&1
cells=$(ls results | grep -E '^[A-E]_.*\.json$' | sed 's/\.json$//')
$PY replay.py $cells > results/replay_all.txt 2>&1
MSG="Results and analysis output, pushed by the unattended run

The analysis_*.txt and replay_all.txt files are the committed analysis
scripts' output. Nobody has read them yet; the README does not cover them."
git add -A results && git commit -q -m "$MSG"
WT="$1"
[ -n "$WT" ] && [ -d "$WT/studies/scaling_curve" ] || { echo "no worktree given or found: local commit only"; exit 0; }
cp ./*.py ./*.md ./*.sh replay_ABC.txt "$WT/studies/scaling_curve/"
cp results/* "$WT/studies/scaling_curve/results/"
cd "$WT" && git add studies/scaling_curve && git commit -q -m "scaling_curve: $MSG" && git push origin scaling-curve-study
