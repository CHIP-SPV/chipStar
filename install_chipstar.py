#!/usr/bin/env python3
"""
CHIP-SPV Suite Installer

Interactive installer for chipStar and all dependent libraries.

Usage:
    # Interactive mode (TUI menu)
    ./install_chipstar.py
    
    # CLI mode (non-interactive)
    ./install_chipstar.py --install-dir ~/install/HIP --components chipstar,rocprim,hipcub
    
    # Install all components
    ./install_chipstar.py --all --install-dir ~/install/HIP
    
    # List available components
    ./install_chipstar.py --list
"""

import argparse
import os
import shutil
import subprocess
import sys
import tty
import termios
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# Python 3.6 compatibility check
if sys.version_info < (3, 6):
    print("Error: Python 3.6 or higher is required")
    sys.exit(1)


# ============================================================================
# Component Definitions
# ============================================================================

class Component:
    """Represents an installable component.

    Build behavior is encoded in data rather than bespoke build_* methods:
      compiler: "llvm" (chipStar), "hipcc", or "icpx".
      cmake_flags: extra -D flags appended to the cmake command.
      test_cmake_flags: -D flags appended only when --with-tests is passed.
      with_clang_compiler_path: append -DCLANG_COMPILER_PATH=<llvm_clang>.
      with_hip_include_flag: append -DCMAKE_CXX_FLAGS=-I<HIP_PATH>/include.
      use_cwd_if_chipstar_repo / git_submodule_update: chipStar-only source
                       handling (reuse in-tree checkout, init submodules).
    """
    def __init__(self, name, display_name, repo, branch="main", depends_on=None,
                 description="", enabled=True,
                 compiler="hipcc",
                 cmake_flags=None,
                 test_cmake_flags=None,
                 with_clang_compiler_path=False,
                 with_hip_include_flag=False,
                 use_cwd_if_chipstar_repo=False,
                 git_submodule_update=False):
        self.name = name
        self.display_name = display_name
        self.repo = repo
        self.branch = branch
        self.depends_on = depends_on if depends_on is not None else []
        self.description = description
        self.enabled = enabled
        self.compiler = compiler
        self.cmake_flags = list(cmake_flags) if cmake_flags else []
        self.test_cmake_flags = list(test_cmake_flags) if test_cmake_flags else []
        self.with_clang_compiler_path = with_clang_compiler_path
        self.with_hip_include_flag = with_hip_include_flag
        self.use_cwd_if_chipstar_repo = use_cwd_if_chipstar_repo
        self.git_submodule_update = git_submodule_update
    
    def __hash__(self):
        return hash(self.name)


# All available components with their dependencies
COMPONENTS = [
    Component(
        name="chipstar",
        display_name="chipStar",
        repo="git@github.com:CHIP-SPV/chipStar.git",
        description="Core HIP runtime for SPIR-V (required)",
        compiler="llvm",
        use_cwd_if_chipstar_repo=True,
        git_submodule_update=True,
    ),
    Component(
        name="rocprim",
        display_name="rocPRIM",
        repo="git@github.com:CHIP-SPV/rocPRIM.git",
        depends_on=["chipstar"],
        description="Parallel primitives library",
        cmake_flags=["-DBUILD_BENCHMARK=OFF", "-DBUILD_TEST=OFF"],
        test_cmake_flags=["-DBUILD_TEST=ON"],
    ),
    Component(
        name="hipcub",
        display_name="hipCUB",
        repo="git@github.com:CHIP-SPV/hipCUB.git",
        depends_on=["chipstar", "rocprim"],
        description="CUB-like primitives for HIP",
        cmake_flags=["-DBUILD_TEST=OFF"],
        test_cmake_flags=["-DBUILD_TEST=ON"],
    ),
    Component(
        name="rocthrust",
        display_name="rocThrust",
        repo="git@github.com:CHIP-SPV/rocThrust.git",
        depends_on=["chipstar", "rocprim"],
        description="Thrust parallel algorithms",
        cmake_flags=["-DBUILD_TEST=OFF"],
        test_cmake_flags=["-DBUILD_TEST=ON"],
    ),
    Component(
        name="rocrand",
        display_name="rocRAND",
        repo="git@github.com:CHIP-SPV/rocRAND.git",
        depends_on=["chipstar"],
        description="Random number generation",
        cmake_flags=[
            "-DBUILD_BENCHMARK=OFF",
            "-DBUILD_TEST=OFF",
            "-DROCRAND_HAVE_ASM_INCBIN=OFF",  # Disable ASM for SPIR-V
        ],
        test_cmake_flags=["-DBUILD_TEST=ON"],
    ),
    Component(
        name="hiprand",
        display_name="hipRAND",
        repo="git@github.com:CHIP-SPV/hipRAND.git",
        depends_on=["chipstar", "rocrand"],
        description="HIP random number interface",
        cmake_flags=["-DBUILD_TEST=OFF"],
        test_cmake_flags=["-DBUILD_TEST=ON"],
    ),
    Component(
        name="rocsparse",
        display_name="rocSPARSE",
        repo="git@github.com:CHIP-SPV/rocSPARSE.git",
        depends_on=["chipstar"],
        description="Sparse matrix operations",
        cmake_flags=["-DBUILD_CLIENTS_SAMPLES=OFF", "-DBUILD_CLIENTS_TESTS=OFF"],
        test_cmake_flags=["-DBUILD_CLIENTS_TESTS=ON"],
    ),
    Component(
        name="hipsparse",
        display_name="hipSPARSE",
        repo="git@github.com:CHIP-SPV/hipSPARSE.git",
        depends_on=["chipstar", "rocsparse"],
        description="HIP sparse matrix interface",
        cmake_flags=["-DBUILD_CLIENTS_SAMPLES=OFF", "-DBUILD_CLIENTS_TESTS=OFF"],
        test_cmake_flags=["-DBUILD_CLIENTS_TESTS=ON"],
    ),
    Component(
        name="mklshim",
        display_name="H4I-MKLShim",
        repo="git@github.com:CHIP-SPV/H4I-MKLShim.git",
        depends_on=["chipstar"],
        description="Intel MKL shim layer",
        compiler="icpx",
    ),
    Component(
        name="hipblas",
        display_name="H4I-HipBLAS",
        repo="git@github.com:CHIP-SPV/H4I-HipBLAS.git",
        depends_on=["chipstar", "mklshim"],
        description="HIP BLAS via MKL",
        cmake_flags=["-DBUILD_SAMPLES=OFF"],
        with_clang_compiler_path=True,
        with_hip_include_flag=True,
    ),
    Component(
        name="hipsolver",
        display_name="H4I-HipSOLVER",
        repo="git@github.com:CHIP-SPV/H4I-HipSOLVER.git",
        depends_on=["chipstar", "mklshim"],
        description="HIP linear solver via MKL",
        cmake_flags=["-DBUILD_SAMPLES=OFF", "-DBUILD_TESTING=OFF"],
        test_cmake_flags=["-DBUILD_TESTING=ON"],
        with_clang_compiler_path=True,
    ),
    Component(
        name="hipfft",
        display_name="H4I-HipFFT",
        repo="git@github.com:CHIP-SPV/H4I-HipFFT.git",
        depends_on=["chipstar", "mklshim"],
        description="HIP FFT via MKL",
    ),
    Component(
        name="chipfft",
        display_name="chipFFT",
        repo="git@github.com:CHIP-SPV/chipFFT.git",
        depends_on=["chipstar"],
        description=("Portable hipFFT API on chipStar + VkFFT (Level Zero). "
                     "Mutually exclusive with H4I-HipFFT — both install "
                     "lib/libhipfft.so. Disabled by default; opt in via "
                     "-c chipfft."),
        enabled=False,  # opt-in; conflicts with H4I-HipFFT
        cmake_flags=[
            "-DCHIPFFT_BUILD_TESTS=OFF",
            "-DCHIPFFT_BACKEND=LEVEL_ZERO",
        ],
        test_cmake_flags=["-DCHIPFFT_BUILD_TESTS=ON"],
        git_submodule_update=True,  # third_party/VkFFT is a submodule
    ),
    Component(
        name="hipmm",
        display_name="hipMM",
        repo="git@github.com:CHIP-SPV/hipMM.git",
        depends_on=["chipstar", "rocprim", "rocthrust", "hipcub"],
        description="HIP memory manager (RMM port)",
        cmake_flags=["-DBUILD_BENCHMARKS=OFF", "-DBUILD_TESTS=OFF"],
        test_cmake_flags=["-DBUILD_TESTS=ON"],
        with_hip_include_flag=True,
    ),
]


# ============================================================================
# Terminal Utilities
# ============================================================================

class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    MAGENTA = '\033[0;35m'
    CYAN = '\033[0;36m'
    WHITE = '\033[1;37m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    NC = '\033[0m'  # No Color


def clear_screen():
    """Clear terminal screen."""
    print('\033[2J\033[H', end='')


def get_key():
    """Get a single keypress from stdin."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
        if ch == '\x1b':
            ch = sys.stdin.read(1)
            if ch == '[':
                ch = sys.stdin.read(1)
                if ch == 'A':
                    return 'up'
                elif ch == 'B':
                    return 'down'
                elif ch == 'C':
                    return 'right'
                elif ch == 'D':
                    return 'left'
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def input_with_default(prompt: str, default: str) -> str:
    """Get input with a default value shown."""
    try:
        result = input(f"{prompt} [{default}]: ").strip()
        return result if result else default
    except (EOFError, KeyboardInterrupt):
        return default


# ============================================================================
# Configuration
# ============================================================================

class InstallConfig:
    """Installation configuration."""
    def __init__(self, install_base=None, module_base=None, staging_dir=None, jobs=None,
                 date_stamp=None, llvm_dir=None, dry_run=False, verbose=True, module_format="tcl",
                 no_install=False, install_only=False, build_tests=False):
        self.install_base = install_base if install_base else Path.home() / "install" / "HIP"
        self.module_base = module_base if module_base else Path.home() / "modulefiles" / "HIP"
        self.staging_dir = staging_dir if staging_dir else Path("/tmp")
        self.jobs = jobs if jobs else (os.cpu_count() or 8)
        self.date_stamp = date_stamp if date_stamp else datetime.now().strftime("%Y.%m.%d")
        self.llvm_dir = llvm_dir
        self.dry_run = dry_run
        self.verbose = verbose
        self.module_format = module_format  # "tcl" or "lua"
        self.no_install = no_install
        self.install_only = install_only
        self.build_tests = build_tests


# ============================================================================
# Interactive Menu System
# ============================================================================

class InteractiveInstaller:
    """Interactive TUI installer."""
    
    def __init__(self, components: List[Component], config: InstallConfig):
        self.components = {c.name: c for c in components}
        self.component_list = list(components)
        self.config = config
        self.cursor_pos = 0
        self.current_menu = "main"  # main, components, paths, confirm
    
    def get_component_by_index(self, idx: int) -> Component:
        return self.component_list[idx]
    
    def resolve_dependencies(self, component: Component) -> List[str]:
        """Get all dependencies recursively."""
        deps = []
        for dep_name in component.depends_on:
            if dep_name in self.components:
                deps.extend(self.resolve_dependencies(self.components[dep_name]))
                deps.append(dep_name)
        return list(dict.fromkeys(deps))  # Preserve order, remove dupes
    
    def enable_with_deps(self, component: Component):
        """Enable a component and all its dependencies."""
        component.enabled = True
        for dep_name in self.resolve_dependencies(component):
            if dep_name in self.components:
                self.components[dep_name].enabled = True
    
    def get_enabled_components(self) -> List[Component]:
        """Get list of enabled components in build order."""
        enabled = [c for c in self.component_list if c.enabled]
        # Sort by dependency order
        result = []
        added = set()
        
        def add_with_deps(comp):
            for dep_name in comp.depends_on:
                if dep_name not in added and dep_name in self.components:
                    add_with_deps(self.components[dep_name])
            if comp.name not in added:
                result.append(comp)
                added.add(comp.name)
        
        for comp in enabled:
            add_with_deps(comp)
        
        return result
    
    def render_header(self):
        """Render the header."""
        print(f"\n{Colors.CYAN}{'=' * 70}{Colors.NC}")
        print(f"{Colors.BOLD}{Colors.WHITE}  CHIP-SPV Suite Installer{Colors.NC}")
        print(f"{Colors.CYAN}{'=' * 70}{Colors.NC}\n")
    
    def render_main_menu(self):
        """Render main menu."""
        clear_screen()
        self.render_header()
        
        print(f"  {Colors.DIM}Use [ARROW] keys to navigate, [ENTER] to select, [CTRL+C] to exit{Colors.NC}\n")
        
        menu_items = [
            ("Select Components", "Choose which libraries to install"),
            ("Configure Paths", "Set installation directories"),
            ("Review & Install", "Review configuration and start installation"),
            ("Exit", "Quit installer"),
        ]
        
        enabled_count = len([c for c in self.component_list if c.enabled])
        
        print(f"  {Colors.GREEN}{enabled_count}{Colors.NC} components selected\n")
        print(f"  {Colors.DIM}Install to: {self.config.install_base}{Colors.NC}\n")
        
        for i, (name, desc) in enumerate(menu_items):
            if i == self.cursor_pos:
                print(f"  {Colors.CYAN}>{Colors.NC} {Colors.BOLD}{name}{Colors.NC}")
                print(f"    {Colors.DIM}{desc}{Colors.NC}")
            else:
                print(f"    {name}")
                print(f"    {Colors.DIM}{desc}{Colors.NC}")
            print()
        
        return len(menu_items)
    
    def render_component_menu(self):
        """Render component selection menu."""
        clear_screen()
        self.render_header()
        
        print(f"  {Colors.DIM}[SPACE] toggle, [A] select all, [N] select none, [ENTER] done{Colors.NC}\n")
        print(f"  {Colors.YELLOW}Note: Dependencies are automatically enabled{Colors.NC}\n")
        
        for i, comp in enumerate(self.component_list):
            is_cursor = i == self.cursor_pos
            checkbox = f"{Colors.GREEN}[X]{Colors.NC}" if comp.enabled else "[ ]"
            
            # Show dependency info
            dep_info = ""
            if comp.depends_on:
                dep_names = ", ".join(comp.depends_on)
                dep_info = f" {Colors.DIM}(requires: {dep_names}){Colors.NC}"
            
            if is_cursor:
                print(f"  {Colors.CYAN}>{Colors.NC} {checkbox} {Colors.BOLD}{comp.display_name}{Colors.NC}{dep_info}")
                print(f"      {Colors.DIM}{comp.description}{Colors.NC}")
            else:
                print(f"    {checkbox} {comp.display_name}{dep_info}")
            
        print(f"\n  {Colors.DIM}Press [ENTER] to return to main menu{Colors.NC}")
        
        return len(self.component_list)
    
    def render_paths_menu(self):
        """Render paths configuration."""
        clear_screen()
        self.render_header()
        
        print(f"  {Colors.DIM}[ENTER] to edit/toggle, [ESC] to go back{Colors.NC}\n")
        
        module_format_display = "Lua" if self.config.module_format == "lua" else "TCL"
        paths = [
            ("Install Directory", str(self.config.install_base)),
            ("Module Directory", str(self.config.module_base)),
            ("Staging Directory", str(self.config.staging_dir)),
            ("Parallel Jobs", str(self.config.jobs)),
            ("Name Postfix", str(self.config.date_stamp)),
            ("Module Format", module_format_display),
            ("< Back to Main Menu", ""),
        ]
        
        for i, (name, value) in enumerate(paths):
            if i == self.cursor_pos:
                if value:
                    print(f"  {Colors.CYAN}>{Colors.NC} {Colors.BOLD}{name}{Colors.NC}")
                    if i == 5:  # Module Format - show as radio button
                        tcl_selected = "(●)" if self.config.module_format == "tcl" else "(○)"
                        lua_selected = "(●)" if self.config.module_format == "lua" else "(○)"
                        print(f"      {Colors.GREEN}{tcl_selected} TCL  {lua_selected} Lua{Colors.NC}")
                    else:
                        print(f"      {Colors.GREEN}{value}{Colors.NC}")
                else:
                    print(f"  {Colors.CYAN}>{Colors.NC} {Colors.BOLD}{name}{Colors.NC}")
            else:
                if value:
                    print(f"    {name}")
                    if i == 5:  # Module Format - show as radio button
                        tcl_selected = "(●)" if self.config.module_format == "tcl" else "(○)"
                        lua_selected = "(●)" if self.config.module_format == "lua" else "(○)"
                        print(f"      {Colors.DIM}{tcl_selected} TCL  {lua_selected} Lua{Colors.NC}")
                    else:
                        print(f"      {Colors.DIM}{value}{Colors.NC}")
                else:
                    print(f"    {name}")
            print()
        
        return len(paths)
    
    def render_confirm_menu(self):
        """Render confirmation screen."""
        clear_screen()
        self.render_header()
        
        enabled = self.get_enabled_components()
        
        print(f"  {Colors.BOLD}Configuration Summary{Colors.NC}\n")
        print(f"  Install Directory: {Colors.GREEN}{self.config.install_base}{Colors.NC}")
        print(f"  Module Directory:  {Colors.GREEN}{self.config.module_base}{Colors.NC}")
        print(f"  Staging Directory: {Colors.GREEN}{self.config.staging_dir}{Colors.NC}")
        print(f"  Parallel Jobs:     {Colors.GREEN}{self.config.jobs}{Colors.NC}")
        print(f"  Name Postfix:      {Colors.GREEN}{self.config.date_stamp}{Colors.NC}")
        print(f"  Module Format:     {Colors.GREEN}{self.config.module_format.upper()}{Colors.NC}")
        print()
        
        print(f"  {Colors.BOLD}Components to Install ({len(enabled)}):{Colors.NC}")
        for i, comp in enumerate(enabled, 1):
            print(f"    {i:2}. {comp.display_name}")
        print()
        
        options = [
            ("Start Installation", "Begin installing selected components"),
            ("Back to Main Menu", "Return to main menu"),
        ]
        
        for i, (name, desc) in enumerate(options):
            if i == self.cursor_pos:
                print(f"  {Colors.CYAN}>{Colors.NC} {Colors.BOLD}{Colors.GREEN if i == 0 else ''}{name}{Colors.NC}")
            else:
                print(f"    {name}")
        
        return len(options)
    
    def edit_path(self, index: int):
        """Edit a path configuration."""
        if index == 5:  # Module Format - toggle between TCL and Lua
            self.config.module_format = "lua" if self.config.module_format == "tcl" else "tcl"
            return
        
        print(f"\n  {Colors.YELLOW}Enter new value (or press Enter to keep current):{Colors.NC}")
        
        if index == 0:
            new_val = input_with_default("  Install Directory", str(self.config.install_base))
            self.config.install_base = Path(new_val)
        elif index == 1:
            new_val = input_with_default("  Module Directory", str(self.config.module_base))
            self.config.module_base = Path(new_val)
        elif index == 2:
            new_val = input_with_default("  Source Directory", str(self.config.staging_dir))
            self.config.staging_dir = Path(new_val)
        elif index == 3:
            new_val = input_with_default("  Parallel Jobs", str(self.config.jobs))
            try:
                self.config.jobs = int(new_val)
            except ValueError:
                pass
        elif index == 4:
            new_val = input_with_default("  Name Postfix", str(self.config.date_stamp))
            if new_val:
                self.config.date_stamp = new_val
    
    def run_main_menu(self) -> Optional[str]:
        """Run main menu interaction."""
        num_items = self.render_main_menu()
        
        key = get_key()
        
        if key == 'up':
            self.cursor_pos = (self.cursor_pos - 1) % num_items
        elif key == 'down':
            self.cursor_pos = (self.cursor_pos + 1) % num_items
        elif key in ('\r', '\n'):
            if self.cursor_pos == 0:
                return "components"
            elif self.cursor_pos == 1:
                return "paths"
            elif self.cursor_pos == 2:
                return "confirm"
            elif self.cursor_pos == 3:
                return "exit"
        elif key == '\x03':  # Ctrl+C
            return "exit"
        
        return "main"
    
    def disable_with_dependents(self, component: Component):
        """Disable a component and all components that depend on it."""
        component.enabled = False
        # Find all components that depend on this one
        for comp in self.component_list:
            if component.name in comp.depends_on and comp.enabled:
                self.disable_with_dependents(comp)
    
    def run_component_menu(self) -> str:
        """Run component selection menu."""
        num_items = self.render_component_menu()
        
        key = get_key()
        
        if key == 'up':
            self.cursor_pos = (self.cursor_pos - 1) % num_items
        elif key == 'down':
            self.cursor_pos = (self.cursor_pos + 1) % num_items
        elif key == ' ':
            comp = self.get_component_by_index(self.cursor_pos)
            if comp.enabled:
                # Disable this component and anything that depends on it
                self.disable_with_dependents(comp)
            else:
                self.enable_with_deps(comp)
        elif key == 'a' or key == 'A':
            for comp in self.component_list:
                comp.enabled = True
        elif key == 'n' or key == 'N':
            for comp in self.component_list:
                comp.enabled = False
        elif key in ('\r', '\n', '\x1b'):
            self.cursor_pos = 0
            return "main"
        elif key == '\x03':  # Ctrl+C
            return "exit"
        
        return "components"
    
    def run_paths_menu(self) -> str:
        """Run paths configuration menu."""
        num_items = self.render_paths_menu()
        
        key = get_key()
        
        if key == 'up':
            self.cursor_pos = (self.cursor_pos - 1) % num_items
        elif key == 'down':
            self.cursor_pos = (self.cursor_pos + 1) % num_items
        elif key in ('\r', '\n'):
            if self.cursor_pos == num_items - 1:  # Back option
                self.cursor_pos = 0
                return "main"
            else:
                self.edit_path(self.cursor_pos)
        elif key == '\x1b':
            self.cursor_pos = 0
            return "main"
        elif key == '\x03':  # Ctrl+C
            return "exit"
        
        return "paths"
    
    def run_confirm_menu(self) -> str:
        """Run confirmation menu."""
        num_items = self.render_confirm_menu()
        
        key = get_key()
        
        if key == 'up':
            self.cursor_pos = (self.cursor_pos - 1) % num_items
        elif key == 'down':
            self.cursor_pos = (self.cursor_pos + 1) % num_items
        elif key in ('\r', '\n'):
            if self.cursor_pos == 0:
                return "install"
            else:
                self.cursor_pos = 0
                return "main"
        elif key == '\x1b':
            self.cursor_pos = 0
            return "main"
        elif key == '\x03':  # Ctrl+C
            return "exit"
        
        return "confirm"
    
    def run(self) -> Optional[List[Component]]:
        """Run the interactive installer.
        
        Returns list of components to install, or None if cancelled.
        """
        self.current_menu = "main"
        self.cursor_pos = 0
        
        while True:
            try:
                if self.current_menu == "main":
                    self.current_menu = self.run_main_menu()
                elif self.current_menu == "components":
                    self.current_menu = self.run_component_menu()
                elif self.current_menu == "paths":
                    self.current_menu = self.run_paths_menu()
                elif self.current_menu == "confirm":
                    self.current_menu = self.run_confirm_menu()
                elif self.current_menu == "install":
                    clear_screen()
                    return self.get_enabled_components()
                elif self.current_menu == "exit":
                    clear_screen()
                    return None
            except KeyboardInterrupt:
                clear_screen()
                return None


# ============================================================================
# Build System
# ============================================================================

class Builder:
    """Handles building and installing components."""
    
    def __init__(self, config: InstallConfig):
        self.config = config
        self.env = os.environ.copy()
        self._components_by_name = {c.name: c for c in COMPONENTS}
        self._setup_environment()
    
    def _setup_environment(self):
        """Set up build environment."""
        # Find LLVM - prefer LLVM_DIR env var, then config, then which clang++
        llvm_dir = None
        
        # Check LLVM_DIR environment variable first
        llvm_dir_env = os.environ.get("LLVM_DIR", "")
        if llvm_dir_env and Path(llvm_dir_env).exists():
            llvm_dir = Path(llvm_dir_env)
        elif self.config.llvm_dir:
            llvm_dir = self.config.llvm_dir
        else:
            # Fall back to which clang++, but try to avoid oneAPI's clang
            clang_path = subprocess.run(
                ["which", "clang++"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
            ).stdout.strip()
            if clang_path:
                potential_dir = Path(clang_path).parent.parent
                # Avoid oneAPI if possible - check for icpx in same dir
                icpx_check = potential_dir / "bin" / "icpx"
                if not icpx_check.exists():
                    llvm_dir = potential_dir
        
        # If still not found, use /usr as fallback
        if not llvm_dir:
            llvm_dir = Path("/usr")
        
        self.llvm_dir = llvm_dir
        self.llvm_clang = llvm_dir / "bin" / "clang++"
        
        # Set CMake generator
        self.env["CMAKE_GENERATOR"] = "Unix Makefiles"
        
        # Detect GCC toolchain for clang/hipcc to use correct libstdc++
        # Only stored here - applied in setup_chipstar_env for hipcc builds
        self.gcc_toolchain = None
        try:
            gcc_path = subprocess.run(
                ["which", "gcc"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
            ).stdout.strip()
            if gcc_path and "/spack/" in gcc_path:
                # Spack-installed GCC - extract toolchain path
                self.gcc_toolchain = str(Path(gcc_path).parent.parent)
        except Exception:
            pass
    
    def print_environment_info(self):
        """Print discovered environment information."""
        print(f"\n{Colors.CYAN}{'=' * 70}{Colors.NC}")
        print(f"{Colors.BOLD}Environment Discovery{Colors.NC}")
        print(f"{Colors.CYAN}{'=' * 70}{Colors.NC}\n")
        
        # LLVM/Clang
        clang_version = "unknown"
        if self.llvm_clang.exists():
            result = subprocess.run(
                [str(self.llvm_clang), "--version"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
            )
            if result.returncode == 0:
                first_line = result.stdout.split('\n')[0]
                clang_version = first_line
        print(f"  {Colors.BOLD}LLVM/Clang:{Colors.NC}")
        print(f"    Path:    {self.llvm_dir}")
        print(f"    Version: {clang_version}")
        
        # Intel oneAPI / icpx
        icpx_path = subprocess.run(["which", "icpx"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True).stdout.strip()
        if icpx_path:
            result = subprocess.run([icpx_path, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            icpx_version = result.stdout.split('\n')[0] if result.returncode == 0 else "unknown"
            print(f"\n  {Colors.BOLD}Intel oneAPI:{Colors.NC}")
            print(f"    icpx:    {icpx_path}")
            print(f"    Version: {icpx_version}")
        else:
            print(f"\n  {Colors.BOLD}Intel oneAPI:{Colors.NC} {Colors.YELLOW}Not found{Colors.NC}")
        
        # MKL - check multiple possible env vars
        mkl_root = os.environ.get("MKLROOT", "") or os.environ.get("MKL_ROOT", "")
        # Also check if oneAPI is loaded - MKL is usually at same path
        if not mkl_root and icpx_path:
            oneapi_root = Path(icpx_path).parent.parent
            mkl_lib = oneapi_root / "lib" / "libmkl_core.so"
            if mkl_lib.exists():
                mkl_root = str(oneapi_root)
        
        if mkl_root:
            print(f"\n  {Colors.BOLD}Intel MKL:{Colors.NC}")
            print(f"    Path:    {mkl_root}")
        else:
            print(f"\n  {Colors.BOLD}Intel MKL:{Colors.NC} {Colors.YELLOW}Not found{Colors.NC}")
        
        # Existing chipStar
        chipstar_path = self._unified_install_dir()
        hipcc_path = chipstar_path / "bin" / "hipcc"
        if hipcc_path.exists():
            print(f"\n  {Colors.BOLD}chipStar (existing):{Colors.NC}")
            print(f"    Path:    {chipstar_path}")
            print(f"    Status:  {Colors.GREEN}Found{Colors.NC}")
        else:
            hip_path_env = os.environ.get("HIP_PATH", "")
            if hip_path_env and Path(hip_path_env).exists():
                print(f"\n  {Colors.BOLD}HIP (from environment):{Colors.NC}")
                print(f"    HIP_PATH: {hip_path_env}")
            else:
                print(f"\n  {Colors.BOLD}chipStar:{Colors.NC} {Colors.YELLOW}Will be built{Colors.NC}")
        
        # Check what's already installed in the unified prefix
        print(f"\n  {Colors.BOLD}Installed Components:{Colors.NC}")
        prefix = self._unified_install_dir()

        # Sentinel paths (relative to the unified prefix) that uniquely
        # indicate a given component has been installed.
        component_sentinels = [
            ("chipStar",      "bin/hipcc"),
            ("rocPRIM",       "include/rocprim/rocprim.hpp"),
            ("hipCUB",        "include/hipcub/hipcub.hpp"),
            ("rocThrust",     "include/thrust/version.h"),
            ("rocRAND",       "include/rocrand/rocrand.h"),
            ("hipRAND",       "include/hiprand/hiprand.h"),
            ("rocSPARSE",     "include/rocsparse/rocsparse.h"),
            ("hipSPARSE",     "include/hipsparse/hipsparse.h"),
            ("H4I-MKLShim",   "lib/libMKLShim.so"),
            ("H4I-HipBLAS",   "lib/libhipblas.so"),
            ("H4I-HipSOLVER", "lib/libhipsolver.so"),
            ("H4I-HipFFT",    "lib/libhipfft.so"),
            ("chipFFT",       "include/chipfft/chipfft_ext.h"),
            ("hipMM",         "include/rmm/rmm.hpp"),
        ]

        for name, sentinel in component_sentinels:
            if (prefix / sentinel).exists():
                print(f"    {Colors.GREEN}[installed]{Colors.NC} {name}")
            else:
                print(f"    {Colors.DIM}[missing]{Colors.NC}   {name}")

        print()
    
    def print_build_plan(self, components: list):
        """Print the build plan with explanations."""
        print(f"\n{Colors.CYAN}{'=' * 70}{Colors.NC}")
        print(f"{Colors.BOLD}Build Plan{Colors.NC}")
        print(f"{Colors.CYAN}{'=' * 70}{Colors.NC}\n")
        
        print(f"  {Colors.BOLD}Target Directory:{Colors.NC} {self._unified_install_dir()}")
        print(f"  {Colors.BOLD}Module Directory:{Colors.NC} {self.config.module_base}")
        print(f"  {Colors.BOLD}Module Format:{Colors.NC} {self.config.module_format.upper()}")
        print(f"  {Colors.BOLD}Staging Directory:{Colors.NC} {self.config.staging_dir}")
        print(f"  {Colors.BOLD}Parallel Jobs:{Colors.NC} {self.config.jobs}")
        print()
        
        print(f"  {Colors.BOLD}Components to Build ({len(components)}):{Colors.NC}\n")
        
        for i, comp in enumerate(components, 1):
            # Determine compiler
            if comp.name == "mklshim":
                compiler = "icpx (Intel SYCL)"
            elif comp.name == "chipstar":
                compiler = "clang++ (LLVM)"
            else:
                compiler = "hipcc (via chipStar)"
            
            # Show dependencies
            deps_str = ""
            if comp.depends_on:
                deps_str = f" -> requires: {', '.join(comp.depends_on)}"
            
            print(f"    {i:2}. {Colors.BOLD}{comp.display_name}{Colors.NC}")
            print(f"        Compiler: {compiler}")
            print(f"        Branch:   {comp.branch}")
            if deps_str:
                print(f"        Deps:     {', '.join(comp.depends_on)}")
            print()
    
    def run_cmd(self, cmd: List[str], cwd: Optional[Path] = None, 
                check: bool = True) -> subprocess.CompletedProcess:
        """Run a command."""
        if self.config.verbose:
            print(f"{Colors.DIM}$ {' '.join(cmd)}{Colors.NC}")
        
        if self.config.dry_run:
            return subprocess.CompletedProcess(cmd, 0)
        
        result = subprocess.run(
            cmd, cwd=cwd, env=self.env,
            check=check
        )
        return result
    
    def clone_or_update_if_needed(self, repo: str, name: str, branch: str, dest: Path):
        """Clone or update unless --install-only (source already present)."""
        if self.config.install_only:
            src_dir = dest / name
            if not src_dir.exists():
                print(f"{Colors.RED}[ERROR]{Colors.NC} --install-only but {src_dir} does not exist")
                sys.exit(1)
            return
        self.clone_or_update(repo, name, branch, dest)

    def clone_or_update(self, repo: str, name: str, branch: str, dest: Path):
        """Clone or update a repository."""
        repo_path = dest / name
        git_dir = repo_path / ".git"
        
        if repo_path.exists() and git_dir.exists():
            # Valid git repo - update it
            print(f"{Colors.YELLOW}[INFO]{Colors.NC} Updating {name}...")
            self.run_cmd(["git", "fetch", "origin"], cwd=repo_path)
            self.run_cmd(["git", "checkout", branch], cwd=repo_path)
            self.run_cmd(["git", "pull", "origin", branch], cwd=repo_path)
        else:
            # Not a git repo or doesn't exist - remove and clone fresh
            if repo_path.exists():
                print(f"{Colors.YELLOW}[INFO]{Colors.NC} Removing invalid {name} directory...")
                shutil.rmtree(repo_path)
            print(f"{Colors.YELLOW}[INFO]{Colors.NC} Cloning {name}...")
            self.run_cmd(["git", "clone", "-b", branch, repo, name], cwd=dest)

    def _prepare_fresh_build_dir(self, build_dir: Path) -> None:
        """Remove and recreate build_dir unless --install-only (reuse existing build tree)."""
        if self.config.install_only:
            return
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)

    def _cmake_configure_and_build(self, build_dir: Path, cmake_args: List[str]) -> None:
        """Run cmake + make unless --install-only."""
        if self.config.install_only:
            return
        self.run_cmd(cmake_args, cwd=build_dir)
        self.run_cmd(["make", f"-j{self.config.jobs}"], cwd=build_dir)

    def _make_install_if_needed(self, build_dir: Path) -> None:
        """Run make install unless --no-install."""
        if not self.config.no_install:
            self.run_cmd(["make", "install"], cwd=build_dir)
    
    def add_to_prefix_path(self, path: Path):
        """Add path to CMAKE_PREFIX_PATH."""
        current = self.env.get("CMAKE_PREFIX_PATH", "")
        if current:
            self.env["CMAKE_PREFIX_PATH"] = f"{path}:{current}"
        else:
            self.env["CMAKE_PREFIX_PATH"] = str(path)
    
    def _unified_install_dir(self) -> Path:
        """All components install into this single prefix."""
        return self.config.install_base / "chipStar" / self.config.date_stamp

    def setup_chipstar_env(self):
        """Set up environment for chipStar-based builds."""
        chipstar_install = self._unified_install_dir()
        
        self.env["HIP_PATH"] = str(chipstar_install)
        self.env["HIP_PLATFORM"] = "spirv"
        
        # Store full path to hipcc (avoid picking up system AMD hipcc)
        self.hipcc_path = str(chipstar_install / "bin" / "hipcc")
        
        # Add to PATH
        path = self.env.get("PATH", "")
        self.env["PATH"] = f"{chipstar_install}/bin:{path}"
        
        # Add to LD_LIBRARY_PATH
        ld_path = self.env.get("LD_LIBRARY_PATH", "")
        self.env["LD_LIBRARY_PATH"] = f"{chipstar_install}/lib:{ld_path}"
        
        # Add GCC toolchain to CXXFLAGS for hipcc (clang) to use correct libstdc++
        if self.gcc_toolchain:
            toolchain_flag = "--gcc-toolchain={}".format(self.gcc_toolchain)
            existing = self.env.get("CXXFLAGS", "")
            if toolchain_flag not in existing:
                self.env["CXXFLAGS"] = "{} {}".format(existing, toolchain_flag).strip()
        
        # Add to CMAKE_PREFIX_PATH
        self.add_to_prefix_path(chipstar_install)
    
    def _is_chipstar_source_tree(self, path: Path) -> bool:
        """Return True if `path` looks like the chipStar source repo (CMakeLists + .git)."""
        cmake = path / "CMakeLists.txt"
        if not cmake.exists() or not (path / ".git").exists():
            return False
        try:
            content = cmake.read_text()
        except (OSError, UnicodeDecodeError):
            return False
        return "project(chipStar" in content or "project(chipstar" in content

    def _resolve_src_dir(self, component: Component) -> Path:
        """Pick the source directory for a component (cwd for in-tree chipStar, else staging)."""
        if component.use_cwd_if_chipstar_repo:
            current_dir = Path.cwd()
            if self._is_chipstar_source_tree(current_dir):
                print(f"{Colors.YELLOW}[INFO]{Colors.NC} Using current chipStar repository: {current_dir}")
                return current_dir
        src_dir = self.config.staging_dir / component.display_name
        self.clone_or_update_if_needed(component.repo, component.display_name,
                                       component.branch, self.config.staging_dir)
        return src_dir

    def _compiler_cmake_flags(self, component: Component) -> List[str]:
        """Return compiler-specific -D flags for the given component."""
        if component.compiler == "llvm":
            return [f"-DLLVM_CONFIG_BIN={self.llvm_dir}/bin/llvm-config"]
        if component.compiler == "icpx":
            icpx_path = subprocess.run(
                ["which", "icpx"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True,
            ).stdout.strip()
            return [
                f"-DCMAKE_CXX_COMPILER={icpx_path}",
                f"-DINTEL_COMPILER_PATH={icpx_path}",
            ]
        # default: hipcc (from chipStar install)
        return [f"-DCMAKE_CXX_COMPILER={self.hipcc_path}"]

    def _build_component(self, component: Component) -> None:
        """Generic build pipeline driven by Component metadata."""
        # Environment: every non-chipStar build consumes the chipStar install.
        # All components share one install prefix, so setup_chipstar_env's
        # CMAKE_PREFIX_PATH entry covers every dependency.
        if component.compiler != "llvm":
            self.setup_chipstar_env()

        src_dir = self._resolve_src_dir(component)
        build_dir = src_dir / "build"
        install_dir = self._unified_install_dir()

        if component.git_submodule_update and not self.config.install_only:
            self.run_cmd(["git", "submodule", "update", "--init", "--recursive"], cwd=src_dir)

        self._prepare_fresh_build_dir(build_dir)

        cmake_args = [
            "cmake", "..",
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            *self._compiler_cmake_flags(component),
        ]
        if component.with_clang_compiler_path:
            cmake_args.append(f"-DCLANG_COMPILER_PATH={self.llvm_clang}")
        cmake_args.extend(component.cmake_flags)
        if self.config.build_tests:
            cmake_args.extend(component.test_cmake_flags)
        if component.with_hip_include_flag:
            hip_path = self.env.get("HIP_PATH", "")
            if hip_path:
                cmake_args.append(f"-DCMAKE_CXX_FLAGS=-I{hip_path}/include")

        self._cmake_configure_and_build(build_dir, cmake_args)
        self._make_install_if_needed(build_dir)
    
    def _generate_module(self, name: str, install_dir: Path, version: Optional[str] = None):
        """Generate a module file (TCL or Lua format)."""
        version = version or self.config.date_stamp
        
        self.config.module_base.mkdir(parents=True, exist_ok=True)
        
        if self.config.module_format == "lua":
            module_file = self.config.module_base / name / f"{version}.lua"
            # Remove parent dir if it exists as a file (legacy flat module file)
            if module_file.parent.exists() and not module_file.parent.is_dir():
                module_file.parent.unlink()
            module_file.parent.mkdir(parents=True, exist_ok=True)
            content = f'''-- -*- lua -*-
local install_dir = "{install_dir}"

prepend_path("CMAKE_PREFIX_PATH", install_dir)
prepend_path("CPATH", install_dir .. "/include")
prepend_path("LD_LIBRARY_PATH", install_dir .. "/lib")
prepend_path("LIBRARY_PATH", install_dir .. "/lib")
prepend_path("PATH", install_dir .. "/bin")
'''
        else:  # TCL format (default)
            module_file = self.config.module_base / name / version
            # Remove parent dir if it exists as a file (legacy flat module file)
            if module_file.parent.exists() and not module_file.parent.is_dir():
                module_file.parent.unlink()
            module_file.parent.mkdir(parents=True, exist_ok=True)
            content = f'''#%Module1.0
##
## {name} {version}
##

set install_dir {install_dir}

prepend-path CMAKE_PREFIX_PATH $install_dir
prepend-path CPATH $install_dir/include
prepend-path LD_LIBRARY_PATH $install_dir/lib
prepend-path LIBRARY_PATH $install_dir/lib
prepend-path PATH $install_dir/bin
'''
        
        if not self.config.dry_run:
            # Remove existing module file/directory if it exists
            if module_file.exists():
                if module_file.is_dir():
                    shutil.rmtree(module_file)
                else:
                    module_file.unlink()
            module_file.write_text(content)
        
        print(f"{Colors.GREEN}[INFO]{Colors.NC} Generated module: {module_file}")
    
    def generate_combined_module(self) -> None:
        """Generate one module file pointing at the unified install prefix."""
        self._generate_module("chipStar", self._unified_install_dir())

    def build(self, component: Component):
        """Build a component."""
        print(f"\n{Colors.CYAN}{'=' * 70}{Colors.NC}")
        print(f"{Colors.BOLD}Building {component.display_name}{Colors.NC}")
        print(f"{Colors.CYAN}{'=' * 70}{Colors.NC}\n")

        if component.name not in self._components_by_name:
            print(f"{Colors.RED}[ERROR]{Colors.NC} Unknown component: {component.name}")
            return
        self._build_component(component)
        print(f"\n{Colors.GREEN}[SUCCESS]{Colors.NC} {component.display_name} installed!")


# ============================================================================
# CLI Interface
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CHIP-SPV Suite Installer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  ./install_chipstar.py
  
  # Install specific components
  ./install_chipstar.py --components chipstar,rocprim,hipcub
  
  # Install all components
  ./install_chipstar.py --all
  
  # Custom install directory
  ./install_chipstar.py --all --install-dir ~/my-hip-install
  
  # Dry run (show what would be done)
  ./install_chipstar.py --all --dry-run
"""
    )
    
    parser.add_argument(
        "--list", action="store_true",
        help="List all available components"
    )
    
    parser.add_argument(
        "--all", action="store_true",
        help="Install all components"
    )
    
    parser.add_argument(
        "--components", "-c", type=str,
        help="Comma-separated list of components to install"
    )
    
    parser.add_argument(
        "--install-dir", "-i", type=str,
        help="Installation base directory (default: ~/install/HIP)"
    )
    
    parser.add_argument(
        "--module-dir", "-m", type=str,
        help="Module files directory (default: ~/modulefiles/HIP)"
    )
    
    parser.add_argument(
        "--staging-dir", "-s", type=str,
        help="Staging directory for cloning and building (default: /tmp)"
    )
    
    parser.add_argument(
        "--jobs", "-j", type=int, default=os.cpu_count(),
        help=f"Number of parallel build jobs (default: {os.cpu_count()})"
    )
    
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Show what would be done without actually doing it"
    )
    
    parser.add_argument(
        "--module-format", choices=["tcl", "lua"], default="tcl",
        help="Module file format: tcl (default) or lua"
    )
    
    parser.add_argument(
        "--verbose", "-v", action="store_true", default=True,
        help="Verbose output"
    )

    parser.add_argument(
        "--no-install", action="store_true",
        help="Build only, skip 'make install' step (for CI presubmit)"
    )

    parser.add_argument(
        "--install-only", action="store_true",
        help="Run 'make install' from existing build dirs, skip configure+build (for post-merge)"
    )

    parser.add_argument(
        "--with-tests", action="store_true",
        help="Build each library's test suite (off by default; CI test stages should set this)"
    )

    return parser.parse_args()


def list_components():
    """Print list of available components."""
    print(f"\n{Colors.BOLD}Available Components:{Colors.NC}\n")
    
    for comp in COMPONENTS:
        deps = f" (requires: {', '.join(comp.depends_on)})" if comp.depends_on else ""
        print(f"  {Colors.CYAN}{comp.name:12}{Colors.NC} {comp.display_name:16} {comp.description}{deps}")
    
    print()


def main():
    args = parse_args()
    
    # List mode
    if args.list:
        list_components()
        return 0
    
    if args.no_install and args.install_only:
        print(f"{Colors.RED}[ERROR]{Colors.NC} --no-install and --install-only are mutually exclusive.")
        return 1

    # Create config
    config = InstallConfig(
        jobs=args.jobs,
        dry_run=args.dry_run,
        verbose=args.verbose,
        module_format=args.module_format,
        no_install=args.no_install,
        install_only=args.install_only,
        build_tests=args.with_tests,
    )
    
    if args.install_dir:
        config.install_base = Path(args.install_dir).expanduser()
    if args.module_dir:
        config.module_base = Path(args.module_dir).expanduser()
    if args.staging_dir:
        config.staging_dir = Path(args.staging_dir).expanduser()
    
    # Determine components to install
    components_to_install = []
    
    if args.all and args.components:
        print(f"{Colors.YELLOW}[WARNING]{Colors.NC} Both --all and -c specified. Using -c (ignoring --all).")
        args.all = False
    
    if args.all:
        # Enable all components
        for comp in COMPONENTS:
            comp.enabled = True
        components_to_install = COMPONENTS[:]
    elif args.components:
        # Parse component list
        component_names = [c.strip() for c in args.components.split(",")]
        component_map = {c.name: c for c in COMPONENTS}
        
        # First disable all components, then enable only requested ones + deps
        for comp in COMPONENTS:
            comp.enabled = False
        
        # Enable requested components and their dependencies
        installer = InteractiveInstaller(COMPONENTS, config)
        for name in component_names:
            if name in component_map:
                installer.enable_with_deps(component_map[name])
            else:
                print(f"{Colors.RED}[ERROR]{Colors.NC} Unknown component: {name}")
                print(f"Use --list to see available components.")
                return 1
        
        components_to_install = installer.get_enabled_components()
    else:
        # Interactive mode
        installer = InteractiveInstaller(COMPONENTS, config)
        result = installer.run()
        
        if result is None:
            print("Installation cancelled.")
            return 0
        
        components_to_install = result
        config = installer.config  # May have been modified
    
    if not components_to_install:
        print("No components selected for installation.")
        return 0
    
    # Create directories
    config.install_base.mkdir(parents=True, exist_ok=True)
    config.module_base.mkdir(parents=True, exist_ok=True)
    config.staging_dir.mkdir(parents=True, exist_ok=True)
    
    # Build
    builder = Builder(config)
    
    # Show environment and build plan
    builder.print_environment_info()
    builder.print_build_plan(components_to_install)
    
    # Confirm before proceeding (unless --all or -c was specified)
    if not args.all and not args.components:
        try:
            response = input(f"  {Colors.BOLD}Proceed with installation? [Y/n]:{Colors.NC} ").strip().lower()
            if response and response not in ('y', 'yes'):
                print("Installation cancelled.")
                return 0
        except (EOFError, KeyboardInterrupt):
            print("\nInstallation cancelled.")
            return 0
    
    print(f"\n{Colors.BOLD}Starting installation of {len(components_to_install)} components...{Colors.NC}\n")
    
    failed = []
    succeeded = []
    
    for comp in components_to_install:
        try:
            builder.build(comp)
            succeeded.append(comp)
        except subprocess.CalledProcessError as e:
            print(f"{Colors.RED}[FAILED]{Colors.NC} {comp.display_name}: {e}")
            failed.append(comp)
        except Exception as e:
            print(f"{Colors.RED}[FAILED]{Colors.NC} {comp.display_name}: {e}")
            failed.append(comp)
    
    if succeeded:
        builder.generate_combined_module()

    # Summary
    print(f"\n{Colors.CYAN}{'=' * 70}{Colors.NC}")
    print(f"{Colors.BOLD}Installation Summary{Colors.NC}")
    print(f"{Colors.CYAN}{'=' * 70}{Colors.NC}\n")
    
    if succeeded:
        print(f"{Colors.GREEN}Succeeded ({len(succeeded)}):{Colors.NC}")
        for comp in succeeded:
            print(f"  - {comp.display_name}")
    
    if failed:
        print(f"\n{Colors.RED}Failed ({len(failed)}):{Colors.NC}")
        for comp in failed:
            print(f"  - {comp.display_name}")
    
    print(f"\nInstall directory: {config.install_base}")
    print(f"Module directory:  {config.module_base}")
    
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
