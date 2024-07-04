#!/bin/bash

# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

###############################################################################
# Utility functions #
#####################
# This script contains utility functions and global variables.
###############################################################################
# Usage:
# Asuming this script is in the same directory:
#   source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"
#   info "Hello World"
###############################################################################


# logging
function info {
  echo -en "[\e[0;32mINFO\e[0m]  "
  info2 "$@"
}
function info2 {
  echo "$@"
}
function warn {
  echo -en "[\e[0;33mWARN\e[0m]  "
  warn2 "$@"
}
function warn2 {
  echo "$@"
}
function error {
  echo -en "[\e[0;31mERROR\e[0m] " >&2
  error2 "$@"
}
function error2 {
  echo "$@" >&2
}
function fatal {
  echo -en "[\e[0;41mFATAL\e[0m] " >&2
  echo "$@" >&2
  exit 2
}


# appends a command to a trap
#
# - 1st arg:  code to add
# - remaining args:  names of traps to modify
#
# Source: https://stackoverflow.com/a/7287873
trap_add() {
  trap_add_cmd=$1; shift || fatal "${FUNCNAME[0]} usage error"
  for trap_add_name in "$@"; do
    trap -- "$(
      # helper fn to get existing trap command from output of trap -p
      # shellcheck disable=SC2317
      extract_trap_cmd() { printf '%s\n' "$3"; }
      # print existing trap command with newline
      eval "extract_trap_cmd $(trap -p "${trap_add_name}")"
      # print the new trap command
      printf '%s\n' "${trap_add_cmd}"
    )" "${trap_add_name}" \
      || fatal "unable to add to trap ${trap_add_name}"
  done
}
# set the trace attribute for the above function.  this is
# required to modify DEBUG or RETURN traps because functions don't
# inherit them unless the trace attribute is set
declare -f -t trap_add

# change working diretory to project root but return later
trap_add "cd $(pwd)" EXIT SIGINT SIGQUIT SIGABRT SIGTERM
cd "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")" || exit 99
