#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage: $0 -D <base_dataset> -t <temp_K> -n <time_ns> [-p <solid|liquid|both>] [-T <train_seeds>] [-V <val_seeds>] [-k <skip_frames>] [-b <bindir>] [-e <element>]
Examples:
  $0 -D m_Na365 -t 365 -n 6
  $0 -D m_Na365 -t 365 -n 6 -p liquid
  $0 -D m_Al933 -t 933 -n 6 -e Al
USAGE
}

base=""
temp=""
time_ns=""
phase="both"
train_seeds="seeds/seeds_train"
val_seeds="seeds/seeds_val"
skip_frames=200
bindir="$(dirname "$0")"
element="Na"

while getopts "D:t:n:p:T:V:k:b:e:h" opt; do
  case $opt in
    D) base="$OPTARG";;
    t) temp="$OPTARG";;
    n) time_ns="$OPTARG";;
    p) phase="$OPTARG";;
    T) train_seeds="$OPTARG";;
    V) val_seeds="$OPTARG";;
    k) skip_frames="$OPTARG";;
    b) bindir="$OPTARG";;
    e) element="$OPTARG";;
    h) usage; exit 0;;
    *) usage; exit 1;;
  esac
done

if [ -z "${base}" ] || [ -z "${temp}" ] || [ -z "${time_ns}" ]; then
  usage; exit 1
fi

# Convert bindir to absolute path
bindir="$(cd "$bindir" && pwd)"

do_phase() {
  local PH="$1"
  local TAG=$( [ "$PH" = "solid" ] && echo "S" || echo "L" )
  "$bindir/run_phase.sh" -d "${base}_${TAG}_train" -p "$PH" -t "${temp}" -n "${time_ns}" -s "${train_seeds}" -o "coordinates.h5" -k "${skip_frames}" -b "$bindir" -e "${element}"
  "$bindir/run_phase.sh" -d "${base}_${TAG}_val"   -p "$PH" -t "${temp}" -n "${time_ns}" -s "${val_seeds}"   -o "coordinates.h5" -k "${skip_frames}" -b "$bindir" -e "${element}"
}

case "$phase" in
  solid)  do_phase solid ;;
  liquid) do_phase liquid ;;
  both)   do_phase solid; do_phase liquid ;;
  *) echo "Unknown -p '$phase' (use solid|liquid|both)"; exit 2;;
esac

echo "✔ All done."
