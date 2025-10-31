"""
uv run src/spectrum/diverse_valid/task_registry.py
"""

import importlib
import inspect
import os
import sys


def _register_task(name: str, task_factory, verbose: bool = True):
    """Register a task with basic validation"""
    try:
        task = task_factory()
        if not hasattr(task, "validate"):
            raise TypeError(f"Task {name} must have a validate method")

        # Only test validation if running in verbose mode (as main script)
        if verbose:
            task.validate("")  # Test basic validation functionality

        return task_factory
    except Exception as e:
        if verbose:
            print(f"Warning: Failed to register task '{name}': {e}")
        return None


def _discover_tasks(verbose: bool = True):
    """Auto-discover task factory functions from tasks/ directory"""
    discovered_tasks = {}

    # Handle path resolution for both script and import contexts
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tasks_dir = os.path.join(script_dir, "tasks")

    # Ensure the parent directory is in sys.path for imports
    parent_dir = os.path.dirname(
        os.path.dirname(script_dir)
    )  # Go up to bayesbench root
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    # Find all Python files in tasks/ directory
    for filename in os.listdir(tasks_dir):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = filename[:-3]  # Remove .py extension

            try:
                # Always use absolute import path
                # module = importlib.import_module(f'eval_scripts.diverse_valid.tasks.{module_name}')
                module = importlib.import_module(
                    f"spectrum.diverse_valid.tasks.{module_name}"
                )

                # Find all functions that return GenerationTask instances
                for name, obj in inspect.getmembers(module, inspect.isfunction):
                    # Skip private functions, imported functions, and test functions
                    if (
                        name.startswith("_")
                        or obj.__module__ != module.__name__
                        or name.startswith("test_")
                    ):
                        continue

                    try:
                        # Test if it's a task factory by calling it
                        result = obj()
                        if hasattr(result, "validate"):  # Duck typing check
                            discovered_tasks[name] = obj
                            if verbose:
                                print(
                                    f"  Found task factory: {name} (from {module_name}.py)"
                                )
                    except Exception as e:
                        # Not a task factory or failed to create, skip
                        if verbose:
                            print(f"  Skipping {name} from {module_name}.py: {e}")
                        continue

            except Exception as e:
                if verbose:
                    print(f"Warning: Could not load tasks from {module_name}.py: {e}")

    return discovered_tasks


def get_tasks(verbose: bool = False):
    """Get all available tasks. Use verbose=True for detailed output."""
    if verbose:
        print("Auto-discovering tasks...")

    _discovered_factories = _discover_tasks(verbose=verbose)

    # Register discovered tasks
    tasks = {}
    for name, factory in _discovered_factories.items():
        validated_factory = _register_task(name, factory, verbose=verbose)
        if validated_factory is not None:
            tasks[name] = validated_factory

    if verbose:
        print(
            f"Successfully registered {len(tasks)} tasks: {sorted(list(tasks.keys()))}"
        )

    return tasks


# When imported as module, register tasks quietly
tasks = get_tasks(verbose=False)

# When run as main script, show detailed output
if __name__ == "__main__":
    tasks = get_tasks(verbose=True)
    # iterate through tasks and print out the name, description, examples, and validation function
    for name, task in tasks.items():
        print(f"Task: {name}")
        print(f"Description: {task().description}")
        print(f"Examples: {task().examples}")
        print()
        breakpoint()
