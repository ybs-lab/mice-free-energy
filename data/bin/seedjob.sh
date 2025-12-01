#!/usr/bin/env bash
set -euo pipefail
if [ $# -lt 2 ]; then
  echo "Usage: PHASE={solid|liquid} TEMP=<K> TIME_NS=<int> $0 <seed> <bindir>" >&2
  exit 2
fi

seed="$1"
bindir="$2"

: "${PHASE:?PHASE env var required (solid|liquid)}}"
: "${TEMP:?TEMP env var required (temperature in K)}}"
: "${TIME_NS:?TIME_NS env var required (ns)}}"
element="${ELEMENT:-Na}"

# Convert bindir to absolute path
bindir="$(cd "$bindir" && pwd)"

# Find repo root (directory containing data/)
caller_pwd="$(pwd)"
script_dir="${bindir}"
repo_root=""
# Try going up from script location
if [ -d "${script_dir}/../../data" ]; then
  repo_root="$(cd "${script_dir}/../.." && pwd)"
elif [ -d "${caller_pwd}/data" ]; then
  repo_root="${caller_pwd}"
else
  # Try to find data/ directory by going up from script
  current="${script_dir}"
  while [ "${current}" != "/" ]; do
    if [ -d "${current}/data" ]; then
      repo_root="${current}"
      break
    fi
    current="$(dirname "${current}")"
  done
fi

if [ -z "${repo_root}" ] || [ ! -d "${repo_root}/data" ]; then
  echo "Could not find repo root (directory containing data/)" >&2
  exit 8
fi

data_root="${repo_root}/data"
template_root="${data_root}/sim_templates/${element}"

if [ ! -d "${template_root}" ]; then
  echo "Template directory not found for element '${element}' (${template_root})" >&2
  exit 4
fi

case "$PHASE" in
  solid)
    template_script="${template_root}/createSolidSim${element}.sh"
    ;;
  liquid)
    template_script="${template_root}/createLiquidSim${element}.sh"
    ;;
  *)
    echo "Unknown PHASE='$PHASE' (expected 'solid' or 'liquid')" >&2
    exit 3
    ;;
esac

if [ ! -x "${template_script}" ]; then
  echo "Template script not found or not executable: ${template_script}" >&2
  exit 5
fi

"${template_script}" -t "$TEMP" -p "$PHASE" -s "$seed" -n "$TIME_NS"
