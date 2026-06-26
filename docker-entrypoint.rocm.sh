#!/bin/bash
set -e

# ROCm variant of the entrypoint. Same UID/GID handling as the CPU image, plus
# it grants the runtime user access to the AMD GPU devices (/dev/kfd, /dev/dri)
# by adding their owning groups to appuser BEFORE the gosu switch — gosu resets
# supplementary groups, so docker's --group-add alone is not enough.

PUID=${PUID:-1000}
PGID=${PGID:-1000}

echo "=== Scriberr (ROCm) Container Setup ==="
echo "Requested UID: $PUID, GID: $PGID"

setup_user() {
    local target_uid=$1
    local target_gid=$2
    if [ "$target_uid" != "1000" ] || [ "$target_gid" != "1000" ]; then
        echo "Setting up custom user with UID=$target_uid, GID=$target_gid..."
        if getent group "$target_gid" >/dev/null 2>&1; then
            echo "Group with GID $target_gid already exists, using it"
        else
            groupmod -g "$target_gid" appuser 2>/dev/null || {
                groupadd -g "$target_gid" appgroup
                usermod -g "$target_gid" appuser
            }
        fi
        usermod -u "$target_uid" appuser 2>/dev/null || {
            echo "Warning: Could not change user ID, continuing with existing user"
        }
        chown -R "$target_uid:$target_gid" /app/data 2>/dev/null || true
    else
        echo "Using default user (UID=1000, GID=1000)"
    fi
}

# Add appuser to the groups that own the passed-through GPU devices so it can
# talk to the kernel driver after the gosu switch.
setup_gpu_access() {
    local dev gid gname
    for dev in /dev/kfd /dev/dri/renderD128 /dev/dri/card0 /dev/dri/card1; do
        [ -e "$dev" ] || continue
        gid=$(stat -c '%g' "$dev" 2>/dev/null) || continue
        [ -n "$gid" ] || continue
        gname=$(getent group "$gid" | cut -d: -f1)
        if [ -z "$gname" ]; then
            gname="gpu$gid"
            groupadd -g "$gid" "$gname" 2>/dev/null || true
            gname=$(getent group "$gid" | cut -d: -f1)
        fi
        if [ -n "$gname" ]; then
            usermod -aG "$gname" appuser 2>/dev/null || true
            echo "GPU access: added appuser to group '$gname' (gid $gid) for $dev"
        fi
    done
}

if [ "$(id -u)" = "0" ]; then
    setup_user "$PUID" "$PGID"
    setup_gpu_access

    echo "Setting up data directories..."
    mkdir -p /app/data/uploads /app/data/transcripts /app/data/hf-cache /app/whisperx-env
    chown -R "$PUID:$PGID" /app/data 2>/dev/null || true
    # The whisper_hf venv is pre-warmed and already owned by 1000:1000 at build
    # time; only re-chown it when a non-default UID/GID was requested (avoids a
    # slow recursive chown of the multi-GB venv on every start).
    if [ "$PUID" != "1000" ] || [ "$PGID" != "1000" ]; then
        chown -R "$PUID:$PGID" /app/whisperx-env 2>/dev/null || true
    fi

    echo "=== Setup Complete — starting as appuser (UID=$PUID, GID=$PGID) ==="
    exec gosu appuser "$@"
else
    echo "Running as non-root user UID=$(id -u), GID=$(id -g)"
    mkdir -p /app/data/uploads /app/data/transcripts /app/data/hf-cache /app/whisperx-env 2>/dev/null || true
    echo "=== Setup Complete — starting Scriberr ==="
    exec "$@"
fi
