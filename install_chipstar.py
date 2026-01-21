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
    """Represents an installable component."""
    def __init__(self, name, display_name, repo, branch="main", depends_on=None, description="", enabled=True):
        self.name = name
        self.display_name = display_name
        self.repo = repo
        self.branch = branch
        self.depends_on = depends_on if depends_on is not None else []
        self.description = description
        self.enabled = enabled
    
    def __hash__(self):
        return hash(self.name)


# All available components with their dependencies
COMPONENTS = [
    Component(
        name="chipstar",
        display_name="chipStar",
        repo="git@github.com:CHIP-SPV/chipStar.git",
        description="Core HIP runtime for SPIR-V (required)",
    ),
    Component(
        name="rocprim",
        display_name="rocPRIM",
        repo="git@github.com:CHIP-SPV/rocPRIM.git",
        depends_on=["chipstar"],
        description="Parallel primitives library",
    ),
    Component(
        name="hipcub",
        display_name="hipCUB",
        repo="git@github.com:CHIP-SPV/hipCUB.git",
        depends_on=["chipstar", "rocprim"],
        description="CUB-like primitives for HIP",
    ),
    Component(
        name="rocthrust",
        display_name="rocThrust",
        repo="git@github.com:CHIP-SPV/rocThrust.git",
        depends_on=["chipstar", "rocprim"],
        description="Thrust parallel algorithms",
    ),
    Component(
        name="rocrand",
        display_name="rocRAND",
        repo="git@github.com:CHIP-SPV/rocRAND.git",
        depends_on=["chipstar"],
        description="Random number generation",
    ),
    Component(
        name="hiprand",
        display_name="hipRAND",
        repo="git@github.com:CHIP-SPV/hipRAND.git",
        depends_on=["chipstar", "rocrand"],
        description="HIP random number interface",
    ),
    Component(
        name="rocsparse",
        display_name="rocSPARSE",
        repo="git@github.com:CHIP-SPV/rocSPARSE.git",
        depends_on=["chipstar"],
        description="Sparse matrix operations",
    ),
    Component(
        name="hipsparse",
        display_name="hipSPARSE",
        repo="git@github.com:CHIP-SPV/hipSPARSE.git",
        depends_on=["chipstar", "rocsparse"],
        description="HIP sparse matrix interface",
    ),
    Component(
        name="mklshim",
        display_name="H4I-MKLShim",
        repo="git@github.com:CHIP-SPV/H4I-MKLShim.git",
        depends_on=["chipstar"],
        description="Intel MKL shim layer",
    ),
    Component(
        name="hipblas",
        display_name="H4I-HipBLAS",
        repo="git@github.com:CHIP-SPV/H4I-HipBLAS.git",
        depends_on=["chipstar", "mklshim"],
        description="HIP BLAS via MKL",
    ),
    Component(
        name="hipsolver",
        display_name="H4I-HipSOLVER",
        repo="git@github.com:CHIP-SPV/H4I-HipSOLVER.git",
        depends_on=["chipstar", "mklshim"],
        description="HIP linear solver via MKL",
    ),
    Component(
        name="hipfft",
        display_name="H4I-HipFFT",
        repo="git@github.com:CHIP-SPV/H4I-HipFFT.git",
        depends_on=["chipstar", "mklshim"],
        description="HIP FFT via MKL",
    ),
    Component(
        name="hipmm",
        display_name="hipMM",
        repo="git@github.com:CHIP-SPV/hipMM.git",
        depends_on=["chipstar", "rocprim", "rocthrust", "hipcub"],
        description="HIP memory manager (RMM port)",
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
                 date_stamp=None, llvm_dir=None, dry_run=False, verbose=True, module_format="tcl"):
        self.install_base = install_base if install_base else Path.home() / "install" / "HIP"
        self.module_base = module_base if module_base else Path.home() / "modulefiles" / "HIP"
        self.staging_dir = staging_dir if staging_dir else Path("/tmp")
        self.jobs = jobs if jobs else (os.cpu_count() or 8)
        self.date_stamp = date_stamp if date_stamp else datetime.now().strftime("%Y.%m.%d")
        self.llvm_dir = llvm_dir
        self.dry_run = dry_run
        self.verbose = verbose
        self.module_format = module_format  # "tcl" or "lua"


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
        chipstar_path = self.config.install_base / "chipStar" / self.config.date_stamp
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
        
        # Check what's already installed
        print(f"\n  {Colors.BOLD}Installed Components:{Colors.NC}")
        installed = []
        not_installed = []
        
        component_dirs = {
            "chipStar": f"chipStar/{self.config.date_stamp}",
            "rocPRIM": f"rocPRIM/{self.config.date_stamp}",
            "hipCUB": f"hipCUB/{self.config.date_stamp}",
            "rocThrust": f"rocThrust/{self.config.date_stamp}",
            "rocRAND": f"rocRAND/{self.config.date_stamp}",
            "hipRAND": f"hipRAND/{self.config.date_stamp}",
            "rocSPARSE": f"rocSPARSE/{self.config.date_stamp}",
            "hipSPARSE": f"hipSPARSE/{self.config.date_stamp}",
            "H4I-MKLShim": f"H4I-MKLShim/{self.config.date_stamp}",
            "H4I-HipBLAS": f"H4I-HipBLAS/{self.config.date_stamp}",
            "H4I-HipSOLVER": f"H4I-HipSOLVER/{self.config.date_stamp}",
            "H4I-HipFFT": f"H4I-HipFFT/{self.config.date_stamp}",
            "hipMM": f"hipMM/{self.config.date_stamp}",
        }
        
        for name, subpath in component_dirs.items():
            install_path = self.config.install_base / subpath
            lib_path = install_path / "lib"
            include_path = install_path / "include"
            if lib_path.exists() or include_path.exists():
                installed.append(name)
            else:
                not_installed.append(name)
        
        if installed:
            for name in installed:
                print(f"    {Colors.GREEN}[installed]{Colors.NC} {name}")
        if not_installed:
            for name in not_installed:
                print(f"    {Colors.DIM}[missing]{Colors.NC}   {name}")
        
        print()
    
    def print_build_plan(self, components: list):
        """Print the build plan with explanations."""
        print(f"\n{Colors.CYAN}{'=' * 70}{Colors.NC}")
        print(f"{Colors.BOLD}Build Plan{Colors.NC}")
        print(f"{Colors.CYAN}{'=' * 70}{Colors.NC}\n")
        
        print(f"  {Colors.BOLD}Target Directory:{Colors.NC} {self.config.install_base}")
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
                import shutil
                shutil.rmtree(repo_path)
            print(f"{Colors.YELLOW}[INFO]{Colors.NC} Cloning {name}...")
            self.run_cmd(["git", "clone", "-b", branch, repo, name], cwd=dest)
    
    def add_to_prefix_path(self, path: Path):
        """Add path to CMAKE_PREFIX_PATH."""
        current = self.env.get("CMAKE_PREFIX_PATH", "")
        if current:
            self.env["CMAKE_PREFIX_PATH"] = f"{path}:{current}"
        else:
            self.env["CMAKE_PREFIX_PATH"] = str(path)
    
    def setup_chipstar_env(self):
        """Set up environment for chipStar-based builds."""
        chipstar_install = self.config.install_base / "chipStar" / self.config.date_stamp
        
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
    
    def build_chipstar(self, component: Component):
        """Build chipStar."""
        src_dir = self.config.staging_dir / "chipStar"
        build_dir = src_dir / "build"
        install_dir = self.config.install_base / "chipStar" / self.config.date_stamp
        
        self.clone_or_update(component.repo, "chipStar", component.branch, 
                            self.config.staging_dir)
        
        # Initialize submodules
        self.run_cmd(["git", "submodule", "update", "--init", "--recursive"], 
                    cwd=src_dir)
        
        # Clear build directory to ensure fresh cmake configuration
        if build_dir.exists():
            import shutil
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)
        
        cmake_args = [
            "cmake", "..",
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            f"-DLLVM_CONFIG={self.llvm_dir}/bin/llvm-config",
        ]
        
        self.run_cmd(cmake_args, cwd=build_dir)
        self.run_cmd(["make", f"-j{self.config.jobs}"], cwd=build_dir)
        self.run_cmd(["make", "install"], cwd=build_dir)
        
        # Generate module file
        self._generate_module("chipStar", install_dir)
    
    def build_rocprim(self, component: Component):
        """Build rocPRIM."""
        self.setup_chipstar_env()
        
        src_dir = self.config.staging_dir / "rocPRIM"
        build_dir = src_dir / "build"
        install_dir = self.config.install_base / "rocPRIM" / self.config.date_stamp
        
        self.clone_or_update(component.repo, "rocPRIM", component.branch,
                            self.config.staging_dir)
        
        if build_dir.exists():
            import shutil
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)
        
        cmake_args = [
            "cmake", "..",
            f"-DCMAKE_CXX_COMPILER={self.hipcc_path}",
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            "-DBUILD_TEST=OFF",
            "-DBUILD_BENCHMARK=OFF",
        ]
        
        self.run_cmd(cmake_args, cwd=build_dir)
        self.run_cmd(["make", f"-j{self.config.jobs}"], cwd=build_dir)
        self.run_cmd(["make", "install"], cwd=build_dir)
        
        self._generate_module("rocPRIM", install_dir)
    
    def build_hipcub(self, component: Component):
        """Build hipCUB."""
        self.setup_chipstar_env()
        self.add_to_prefix_path(self.config.install_base / "rocPRIM" / self.config.date_stamp)
        
        src_dir = self.config.staging_dir / "hipCUB"
        build_dir = src_dir / "build"
        install_dir = self.config.install_base / "hipCUB" / self.config.date_stamp
        
        self.clone_or_update(component.repo, "hipCUB", component.branch,
                            self.config.staging_dir)
        
        if build_dir.exists():
            import shutil
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)
        
        cmake_args = [
            "cmake", "..",
            f"-DCMAKE_CXX_COMPILER={self.hipcc_path}",
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            "-DBUILD_TEST=OFF",
        ]
        
        self.run_cmd(cmake_args, cwd=build_dir)
        self.run_cmd(["make", f"-j{self.config.jobs}"], cwd=build_dir)
        self.run_cmd(["make", "install"], cwd=build_dir)
        
        self._generate_module("hipCUB", install_dir)
    
    def build_rocthrust(self, component: Component):
        """Build rocThrust."""
        self.setup_chipstar_env()
        self.add_to_prefix_path(self.config.install_base / "rocPRIM" / self.config.date_stamp)
        
        src_dir = self.config.staging_dir / "rocThrust"
        build_dir = src_dir / "build"
        install_dir = self.config.install_base / "rocThrust" / self.config.date_stamp
        
        self.clone_or_update(component.repo, "rocThrust", component.branch,
                            self.config.staging_dir)
        
        if build_dir.exists():
            import shutil
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)
        
        cmake_args = [
            "cmake", "..",
            f"-DCMAKE_CXX_COMPILER={self.hipcc_path}",
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            "-DBUILD_TEST=OFF",
        ]
        
        self.run_cmd(cmake_args, cwd=build_dir)
        self.run_cmd(["make", f"-j{self.config.jobs}"], cwd=build_dir)
        self.run_cmd(["make", "install"], cwd=build_dir)
        
        self._generate_module("rocThrust", install_dir)
    
    def build_mklshim(self, component: Component):
        """Build H4I-MKLShim."""
        # MKLShim needs HIP headers for the interface
        self.setup_chipstar_env()
        
        src_dir = self.config.staging_dir / "H4I-MKLShim"
        build_dir = src_dir / "build"
        install_dir = self.config.install_base / "H4I-MKLShim" / self.config.date_stamp
        
        self.clone_or_update(component.repo, "H4I-MKLShim", component.branch,
                            self.config.staging_dir)
        
        if build_dir.exists():
            import shutil
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)
        
        # Find icpx
        icpx_path = subprocess.run(
            ["which", "icpx"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
        ).stdout.strip()
        
        cmake_args = [
            "cmake", "..",
            f"-DCMAKE_CXX_COMPILER={icpx_path}",
            f"-DINTEL_COMPILER_PATH={icpx_path}",
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        ]
        
        self.run_cmd(cmake_args, cwd=build_dir)
        self.run_cmd(["make", f"-j{self.config.jobs}"], cwd=build_dir)
        self.run_cmd(["make", "install"], cwd=build_dir)
        
        self._generate_module("H4I-MKLShim", install_dir)
    
    def build_hipblas(self, component: Component):
        """Build H4I-HipBLAS."""
        self.setup_chipstar_env()
        self.add_to_prefix_path(self.config.install_base / "H4I-MKLShim" / self.config.date_stamp)
        
        src_dir = self.config.staging_dir / "H4I-HipBLAS"
        build_dir = src_dir / "build"
        install_dir = self.config.install_base / "H4I-HipBLAS" / self.config.date_stamp
        
        self.clone_or_update(component.repo, "H4I-HipBLAS", component.branch,
                            self.config.staging_dir)
        
        if build_dir.exists():
            import shutil
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)
        
        # Get HIP include path from environment
        hip_path = self.env.get("HIP_PATH", "")
        hip_include = "{}/include".format(hip_path) if hip_path else ""
        
        cmake_args = [
            "cmake", "..",
            f"-DCMAKE_CXX_COMPILER={self.hipcc_path}",
            f"-DCLANG_COMPILER_PATH={self.llvm_clang}",
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            "-DBUILD_SAMPLES=OFF",
        ]
        
        # Add HIP include path to fix 'hip/hip_runtime.h' not found
        if hip_include:
            cmake_args.append(f"-DCMAKE_CXX_FLAGS=-I{hip_include}")
        
        self.run_cmd(cmake_args, cwd=build_dir)
        self.run_cmd(["make", f"-j{self.config.jobs}"], cwd=build_dir)
        self.run_cmd(["make", "install"], cwd=build_dir)
        
        self._generate_module("H4I-HipBLAS", install_dir)
    
    def build_hipsolver(self, component: Component):
        """Build H4I-HipSOLVER."""
        self.setup_chipstar_env()
        self.add_to_prefix_path(self.config.install_base / "H4I-MKLShim" / self.config.date_stamp)
        
        src_dir = self.config.staging_dir / "H4I-HipSOLVER"
        build_dir = src_dir / "build"
        install_dir = self.config.install_base / "H4I-HipSOLVER" / self.config.date_stamp
        
        self.clone_or_update(component.repo, "H4I-HipSOLVER", component.branch,
                            self.config.staging_dir)
        
        if build_dir.exists():
            import shutil
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)
        
        cmake_args = [
            "cmake", "..",
            f"-DCMAKE_CXX_COMPILER={self.hipcc_path}",
            f"-DCLANG_COMPILER_PATH={self.llvm_clang}",
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            "-DBUILD_TESTING=OFF",
            "-DBUILD_SAMPLES=OFF",
        ]
        
        self.run_cmd(cmake_args, cwd=build_dir)
        self.run_cmd(["make", f"-j{self.config.jobs}"], cwd=build_dir)
        self.run_cmd(["make", "install"], cwd=build_dir)
        
        self._generate_module("H4I-HipSOLVER", install_dir)
    
    def build_hipfft(self, component: Component):
        """Build H4I-HipFFT."""
        self.setup_chipstar_env()
        self.add_to_prefix_path(self.config.install_base / "H4I-MKLShim" / self.config.date_stamp)
        
        src_dir = self.config.staging_dir / "H4I-HipFFT"
        build_dir = src_dir / "build"
        install_dir = self.config.install_base / "H4I-HipFFT" / self.config.date_stamp
        
        self.clone_or_update(component.repo, "H4I-HipFFT", component.branch,
                            self.config.staging_dir)
        
        if build_dir.exists():
            import shutil
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)
        
        cmake_args = [
            "cmake", "..",
            f"-DCMAKE_CXX_COMPILER={self.hipcc_path}",
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        ]
        
        self.run_cmd(cmake_args, cwd=build_dir)
        self.run_cmd(["make", f"-j{self.config.jobs}"], cwd=build_dir)
        self.run_cmd(["make", "install"], cwd=build_dir)
        
        self._generate_module("H4I-HipFFT", install_dir)
    
    def build_rocrand(self, component: Component):
        """Build rocRAND."""
        self.setup_chipstar_env()
        
        src_dir = self.config.staging_dir / "rocRAND"
        build_dir = src_dir / "build"
        install_dir = self.config.install_base / "rocRAND" / self.config.date_stamp
        
        self.clone_or_update(component.repo, "rocRAND", component.branch,
                            self.config.staging_dir)
        
        if build_dir.exists():
            import shutil
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)
        
        cmake_args = [
            "cmake", "..",
            f"-DCMAKE_CXX_COMPILER={self.hipcc_path}",
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            "-DBUILD_TEST=OFF",
            "-DBUILD_BENCHMARK=OFF",
            "-DROCRAND_HAVE_ASM_INCBIN=OFF",  # Disable ASM for SPIR-V
        ]
        
        self.run_cmd(cmake_args, cwd=build_dir)
        self.run_cmd(["make", f"-j{self.config.jobs}"], cwd=build_dir)
        self.run_cmd(["make", "install"], cwd=build_dir)
        
        self._generate_module("rocRAND", install_dir)
    
    def build_hiprand(self, component: Component):
        """Build hipRAND."""
        self.setup_chipstar_env()
        self.add_to_prefix_path(self.config.install_base / "rocRAND" / self.config.date_stamp)
        
        src_dir = self.config.staging_dir / "hipRAND"
        build_dir = src_dir / "build"
        install_dir = self.config.install_base / "hipRAND" / self.config.date_stamp
        
        self.clone_or_update(component.repo, "hipRAND", component.branch,
                            self.config.staging_dir)
        
        if build_dir.exists():
            import shutil
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)
        
        cmake_args = [
            "cmake", "..",
            f"-DCMAKE_CXX_COMPILER={self.hipcc_path}",
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            "-DBUILD_TEST=OFF",
        ]
        
        self.run_cmd(cmake_args, cwd=build_dir)
        self.run_cmd(["make", f"-j{self.config.jobs}"], cwd=build_dir)
        self.run_cmd(["make", "install"], cwd=build_dir)
        
        self._generate_module("hipRAND", install_dir)
    
    def build_rocsparse(self, component: Component):
        """Build rocSPARSE."""
        self.setup_chipstar_env()
        
        src_dir = self.config.staging_dir / "rocSPARSE"
        build_dir = src_dir / "build"
        install_dir = self.config.install_base / "rocSPARSE" / self.config.date_stamp
        
        self.clone_or_update(component.repo, "rocSPARSE", component.branch,
                            self.config.staging_dir)
        
        if build_dir.exists():
            import shutil
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)
        
        cmake_args = [
            "cmake", "..",
            f"-DCMAKE_CXX_COMPILER={self.hipcc_path}",
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            "-DBUILD_CLIENTS_TESTS=OFF",
            "-DBUILD_CLIENTS_SAMPLES=OFF",
        ]
        
        self.run_cmd(cmake_args, cwd=build_dir)
        self.run_cmd(["make", f"-j{self.config.jobs}"], cwd=build_dir)
        self.run_cmd(["make", "install"], cwd=build_dir)
        
        self._generate_module("rocSPARSE", install_dir)
    
    def build_hipsparse(self, component: Component):
        """Build hipSPARSE."""
        self.setup_chipstar_env()
        self.add_to_prefix_path(self.config.install_base / "rocSPARSE" / self.config.date_stamp)
        
        src_dir = self.config.staging_dir / "hipSPARSE"
        build_dir = src_dir / "build"
        install_dir = self.config.install_base / "hipSPARSE" / self.config.date_stamp
        
        self.clone_or_update(component.repo, "hipSPARSE", component.branch,
                            self.config.staging_dir)
        
        if build_dir.exists():
            import shutil
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)
        
        cmake_args = [
            "cmake", "..",
            f"-DCMAKE_CXX_COMPILER={self.hipcc_path}",
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            "-DBUILD_CLIENTS_TESTS=OFF",
            "-DBUILD_CLIENTS_SAMPLES=OFF",
        ]
        
        self.run_cmd(cmake_args, cwd=build_dir)
        self.run_cmd(["make", f"-j{self.config.jobs}"], cwd=build_dir)
        self.run_cmd(["make", "install"], cwd=build_dir)
        
        self._generate_module("hipSPARSE", install_dir)
    
    def build_hipmm(self, component: Component):
        """Build hipMM."""
        self.setup_chipstar_env()
        self.add_to_prefix_path(self.config.install_base / "rocPRIM" / self.config.date_stamp)
        self.add_to_prefix_path(self.config.install_base / "hipCUB" / self.config.date_stamp)
        self.add_to_prefix_path(self.config.install_base / "rocThrust" / self.config.date_stamp)
        
        src_dir = self.config.staging_dir / "hipMM"
        build_dir = src_dir / "build"
        install_dir = self.config.install_base / "hipMM" / self.config.date_stamp
        
        self.clone_or_update(component.repo, "hipMM", component.branch,
                            self.config.staging_dir)
        
        if build_dir.exists():
            import shutil
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)
        
        cmake_args = [
            "cmake", "..",
            f"-DCMAKE_CXX_COMPILER={self.hipcc_path}",
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            "-DBUILD_TESTS=OFF",
            "-DBUILD_BENCHMARKS=OFF",
        ]
        
        self.run_cmd(cmake_args, cwd=build_dir)
        self.run_cmd(["make", f"-j{self.config.jobs}"], cwd=build_dir)
        self.run_cmd(["make", "install"], cwd=build_dir)
        
        self._generate_module("hipMM", install_dir)
    
    def _generate_module(self, name: str, install_dir: Path, version: Optional[str] = None):
        """Generate a module file (TCL or Lua format)."""
        version = version or self.config.date_stamp
        
        self.config.module_base.mkdir(parents=True, exist_ok=True)
        
        if self.config.module_format == "lua":
            module_file = self.config.module_base / name / f"{version}.lua"
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
            module_file.write_text(content)
        
        print(f"{Colors.GREEN}[INFO]{Colors.NC} Generated module: {module_file}")
    
    def build(self, component: Component):
        """Build a component."""
        print(f"\n{Colors.CYAN}{'=' * 70}{Colors.NC}")
        print(f"{Colors.BOLD}Building {component.display_name}{Colors.NC}")
        print(f"{Colors.CYAN}{'=' * 70}{Colors.NC}\n")
        
        # Dispatch to appropriate build function
        build_methods = {
            "chipstar": self.build_chipstar,
            "rocprim": self.build_rocprim,
            "hipcub": self.build_hipcub,
            "rocthrust": self.build_rocthrust,
            "mklshim": self.build_mklshim,
            "hipblas": self.build_hipblas,
            "hipsolver": self.build_hipsolver,
            "hipfft": self.build_hipfft,
            "rocrand": self.build_rocrand,
            "hiprand": self.build_hiprand,
            "rocsparse": self.build_rocsparse,
            "hipsparse": self.build_hipsparse,
            "hipmm": self.build_hipmm,
        }
        
        if component.name in build_methods:
            build_methods[component.name](component)
            print(f"\n{Colors.GREEN}[SUCCESS]{Colors.NC} {component.display_name} installed!")
        else:
            print(f"{Colors.RED}[ERROR]{Colors.NC} Unknown component: {component.name}")


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
    
    # Create config
    config = InstallConfig(
        jobs=args.jobs,
        dry_run=args.dry_run,
        verbose=args.verbose,
        module_format=args.module_format,
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
