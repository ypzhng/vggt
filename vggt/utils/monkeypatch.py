import copy
import functools
import types

def copy_func_with_new_globals(f, globals=None):
    """Based on https://stackoverflow.com/a/13503277/2988730 (@unutbu)"""
    if globals is None:
        globals = f.__globals__
    g = types.FunctionType(f.__code__, globals, name=f.__name__,
                           argdefs=f.__defaults__, closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__module__ = f.__module__
    g.__kwdefaults__ = copy.copy(f.__kwdefaults__)
    return g

def add_wrapper_after_function_call_in_method(module, method_name, function_name, wrapper_fn):
    original_method = getattr(module, method_name).__func__
    method_globals = dict(original_method.__globals__)

    # Try global-name replacement first
    target_fn = method_globals.get(function_name, None)
    if target_fn is None:
        # Fallback: treat function_name as attribute on the module (e.g., "rope")
        if hasattr(module, function_name):
            target_fn = getattr(module, function_name)
            wrapper = wrapper_fn(target_fn)
            setattr(module, function_name, wrapper)
            return wrapper
        # If still not found, provide a clearer error
        raise KeyError(f"{function_name} not found as a global in {module.__class__.__name__}.{method_name} "
                       f"and {function_name} not found as attribute on the module.")

    # Original global injection path
    wrapper = wrapper_fn(target_fn)
    method_globals[function_name] = wrapper
    new_method = copy_func_with_new_globals(original_method, globals=method_globals)
    setattr(module, method_name, new_method.__get__(module))
    return wrapper

    