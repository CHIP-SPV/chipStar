#!/bin/bash
# Source this to initialize Environment Modules and register the shared
# modulefiles tree. Prefers a user-local Modules install (5.4+, supports
# pushenv) over the system package, then Lmod, then the system Modules.
#
#   source scripts/module-env.sh
#   module load oneapi/2025.0.4 llvm/22.0-native level-zero/dgpu

if [ -f "$HOME/.local/init/bash" ]; then
  export MODULESHOME=$HOME/.local
  source "$MODULESHOME/init/bash"
elif [ -f /etc/profile.d/lmod.sh ]; then
  source /etc/profile.d/lmod.sh
else
  source /etc/profile.d/modules.sh
fi
module use /space/pvelesko/modulefiles 2>/dev/null || true
module use ~/modulefiles 2>/dev/null || true
