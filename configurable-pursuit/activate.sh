# Source from Bash. Explicit overrides win; otherwise try compatible installations.
_od_activate_release() {
    local release_root activation="${OMNIDRONES_ACTIVATE-}" od_source="${OMNIDRONES_SOURCE-}" prefix
    local -a prefixes=("$HOME/.local/share/wb-platform" "/opt/lab")
    release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || return
    if [[ ${OMNIDRONES_ACTIVATE+x} && ! -f "$activation" ]]; then
        printf 'Invalid explicit OMNIDRONES_ACTIVATE: %s\n' "$activation" >&2; return 1
    fi
    if [[ ${OMNIDRONES_SOURCE+x} && ! -f "$od_source/examples/demo.yaml" ]]; then
        printf 'Invalid explicit OMNIDRONES_SOURCE: %s\n' "$od_source" >&2; return 1
    fi
    # Prefer the known deployment matching an explicit activation/source first.
    for prefix in "${prefixes[@]}"; do
        if [[ "$activation" == "$prefix/apps/activate-omnidrones41.sh" ||
              "$od_source" == "$prefix/src/OmniDrones-isaac41" ]]; then
            prefixes=("$prefix" "${prefixes[@]}"); break
        fi
    done
    for prefix in "${prefixes[@]}"; do
        if [[ -f "${activation:-$prefix/apps/activate-omnidrones41.sh}" &&
              -f "${od_source:-$prefix/src/OmniDrones-isaac41}/examples/demo.yaml" ]]; then
            activation="${activation:-$prefix/apps/activate-omnidrones41.sh}"
            od_source="${od_source:-$prefix/src/OmniDrones-isaac41}"
            break
        fi
    done
    if [[ ! -f "$activation" || ! -f "$od_source/examples/demo.yaml" ]]; then
        printf '%s\n' 'No compatible ROG/WB deployment found. Set OMNIDRONES_ACTIVATE and OMNIDRONES_SOURCE to existing paths.' >&2
        return 1
    fi
    activation="$(cd "$(dirname "$activation")" && pwd)/$(basename "$activation")" || return
    od_source="$(cd "$od_source" && pwd)" || return
    export OMNIDRONES_SOURCE="$od_source"
    source "$activation" || return
    if [[ "$OMNIDRONES_SOURCE" != "$od_source" ]]; then
        printf '%s\n' 'Activation script changed the selected OMNIDRONES_SOURCE; fix the explicit activation script.' >&2
        return 1
    fi
    export OMNIDRONES_ACTIVATE="$activation"
    cd "$release_root"
}
_od_activate_release
_od_activation_status=$?
unset -f _od_activate_release
if (( _od_activation_status )); then unset _od_activation_status; return 1; fi
unset _od_activation_status
