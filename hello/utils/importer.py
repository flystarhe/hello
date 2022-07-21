from importlib import util


def load_from_file(module_name, file_path):
    try:
        spec = util.spec_from_file_location(module_name, file_path)
        module = util.module_from_spec(spec)
        # sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"failure: {e}")

    return module
