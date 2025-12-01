#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage: $0 -d <dataset_name> -p <solid|liquid> -t <temp_K> -n <time_ns> -s <seeds_file> [-o <output.h5>] [-k <skip_frames>] [-b <bindir>] [-R <simulation_root>] [-e <element>]
Examples:
  $0 -d m_Na365_S -p solid -t 365 -n 6 -s seeds/seeds20 -o coordinates.h5 -k 200
  $0 -d m_Al933_S -p solid -t 933 -n 6 -s seeds/seeds20 -o coordinates.h5 -k 200 -e Al
USAGE
}

dataset=""
phase=""
temp=""
time_ns=""
seeds_file=""
output="coordinates.h5"
skip_frames=200
bindir="$(dirname "$0")"
simulation_root=""
element="Na"
caller_pwd="$(pwd)"

while getopts "d:p:t:n:s:o:k:b:R:e:h" opt; do
  case $opt in
    d) dataset="$OPTARG";;
    p) phase="$OPTARG";;
    t) temp="$OPTARG";;
    n) time_ns="$OPTARG";;
    s) seeds_file="$OPTARG";;
    o) output="$OPTARG";;
    k) skip_frames="$OPTARG";;
    b) bindir="$OPTARG";;
    R) simulation_root="$OPTARG";;
    e) element="$OPTARG";;
    h) usage; exit 0;;
    *) usage; exit 1;;
  esac
done

if [ -z "${dataset}" ] || [ -z "${phase}" ] || [ -z "${temp}" ] || [ -z "${time_ns}" ] || [ -z "${seeds_file}" ]; then
  usage; exit 1
fi

# Convert bindir to absolute path
bindir="$(cd "$bindir" && pwd)"

# Find repo root (directory containing data/)
script_dir="$(cd "$(dirname "$0")" && pwd)"
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

# Resolve key directories (data/, simulation/, templates, tools)
data_root="${repo_root}/data"
python_tools_dir="${data_root}/tools/python"
sim_templates_root="${data_root}/sim_templates/${element}"

if [ -z "$simulation_root" ]; then
  simulation_root="${data_root}/simulation"
fi
# Ensure the simulation root directory exists
if [ ! -d "$simulation_root" ]; then
  mkdir -p "$simulation_root"
fi
simulation_root="$(cd "$simulation_root" && pwd)"

dataset_dir="${simulation_root}/${dataset}"
if [ ! -d "${dataset_dir}" ]; then
  mkdir -p "${dataset_dir}"
fi

# ensure helper paths exist
if [ ! -d "$python_tools_dir" ]; then
  echo "Python tools directory not found: ${python_tools_dir}" >&2
  exit 6
fi

if [ ! -d "$sim_templates_root" ]; then
  echo "Simulation templates not found for element '${element}' in ${sim_templates_root}" >&2
  exit 7
fi

# Resolve seeds file to an absolute path (accept relative to data_root or caller cwd)
if [ -f "${seeds_file}" ]; then
  true
elif [ -f "${data_root}/${seeds_file}" ]; then
  seeds_file="${data_root}/${seeds_file}"
elif [ -f "${caller_pwd}/${seeds_file}" ]; then
  seeds_file="${caller_pwd}/${seeds_file}"
else
  echo "Seeds file not found: ${seeds_file}" >&2
  exit 5
fi
seeds_file="$(cd "$(dirname "$seeds_file")" && pwd)/$(basename "$seeds_file")"

# Save resolved seeds_file path before we change directories
resolved_seeds_file="${seeds_file}"

# Copy element-specific plumed file next to dataset
plumed_name="plumed${element}.dat"
plumed_src="${sim_templates_root}/${plumed_name}"
plumed_dst="${dataset_dir}/${plumed_name}"
if [ -f "${plumed_src}" ] && [ ! -f "${plumed_dst}" ]; then
  cp "${plumed_src}" "${plumed_dst}"
elif [ ! -f "${plumed_dst}" ]; then
  echo "${plumed_name} not found (looked in ${plumed_src})" >&2
fi

pushd "${dataset_dir}" >/dev/null

export PHASE="${phase}"
export TEMP="${temp}"
export TIME_NS="${time_ns}"
export ELEMENT="${element}"

XARGS_P="${XARGS_P:-}"

# Run per-seed jobs
cat "${seeds_file}" | xargs ${XARGS_P} -I {} bash "${bindir}/seedjob.sh" {} "${bindir}"
echo "Finished running jobs for all seeds in phase '${phase}'."

popd >/dev/null
# Pack to HDF5 (group per seed)
python3 "${python_tools_dir}/pack_coordinates.py" -d "${dataset_dir}" -s "${resolved_seeds_file}" -o "${dataset_dir}/${output}" -p "${phase}" -t "${temp}" -k "${skip_frames}"
echo "Phase '${phase}' dataset built and packed: ${dataset_dir}/${output}"
