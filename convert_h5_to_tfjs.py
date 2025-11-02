import sys
import numpy as np

# Add deprecated NumPy aliases that older packages still reference
# This avoids editing site-packages directly.
for _name, _val in (('object', object), ('bool', bool), ('int', int), ('float', float)):
    if not hasattr(np, _name):
        setattr(np, _name, _val)

def main():
    # Build sys.argv as if called from the tensorflowjs_converter CLI
    sys.argv = ['tensorflowjs_converter', '--input_format=keras', 'oral_cancer_model.h5', 'tfjs_model']

    # Monkey-patch tensorflow compat APIs that some tensorflowjs dependencies expect
    try:
        import tensorflow as tf
        # Ensure compat.v1.estimator exists with an Exporter attribute so
        # tensorflow_hub imports don't fail when they reference it.
        if not hasattr(tf, 'compat'):
            tf.compat = type('compat', (), {})()
        if not hasattr(tf.compat, 'v1'):
            tf.compat.v1 = type('v1', (), {})()
        if not hasattr(tf.compat.v1, 'estimator'):
            import types
            # Minimal stub with Exporter base class to satisfy isinstance/subclass checks
            stub = types.SimpleNamespace(Exporter=object)
            setattr(tf.compat.v1, 'estimator', stub)
    except Exception:
        # If TensorFlow isn't importable here, the converter will import it later and fail;
        # we ignore errors at this point and rely on the converter to report them.
        pass

    try:
        # Load the keras_h5_conversion module directly from the site-packages
        # file to avoid executing the tensorflowjs package __init__ which
        # imports modules that pull in tensorflow_hub (and cause import
        # time errors). This uses importlib to load by file path.
        import importlib.util
        import os
        import types
        site_pkg_path = os.path.join(sys.prefix, 'Lib', 'site-packages')
        base_tfjs = os.path.join(site_pkg_path, 'tensorflowjs')
        kh5_path = os.path.join(base_tfjs, 'converters', 'keras_h5_conversion.py')
        if not os.path.exists(kh5_path):
            raise FileNotFoundError('Could not find keras_h5_conversion at: ' + kh5_path)

        sys_modules = sys.modules
        tfjs_pkg = types.ModuleType('tensorflowjs')
        # Register fake package before loading submodules so their imports
        # reference this stub package instead of executing the real one.
        sys_modules['tensorflowjs'] = tfjs_pkg

        # Load quantization
        quant_path = os.path.join(base_tfjs, 'quantization.py')
        if not os.path.exists(quant_path):
            raise FileNotFoundError('Missing quantization module: ' + quant_path)
        spec_quant = importlib.util.spec_from_file_location('tensorflowjs.quantization', quant_path)
        mod_quant = importlib.util.module_from_spec(spec_quant)
        spec_quant.loader.exec_module(mod_quant)
        setattr(tfjs_pkg, 'quantization', mod_quant)
        sys_modules['tensorflowjs.quantization'] = mod_quant

        # Load read_weights (depends on quantization)
        read_path = os.path.join(base_tfjs, 'read_weights.py')
        spec_read = importlib.util.spec_from_file_location('tensorflowjs.read_weights', read_path)
        mod_read = importlib.util.module_from_spec(spec_read)
        spec_read.loader.exec_module(mod_read)
        setattr(tfjs_pkg, 'read_weights', mod_read)
        sys_modules['tensorflowjs.read_weights'] = mod_read

        # Load write_weights (depends on read_weights and quantization)
        write_path = os.path.join(base_tfjs, 'write_weights.py')
        spec_write = importlib.util.spec_from_file_location('tensorflowjs.write_weights', write_path)
        mod_write = importlib.util.module_from_spec(spec_write)
        spec_write.loader.exec_module(mod_write)
        setattr(tfjs_pkg, 'write_weights', mod_write)
        sys_modules['tensorflowjs.write_weights'] = mod_write

        # Preload package-level version module so converters.common can use it
        version_path = os.path.join(base_tfjs, 'version.py')
        spec_version = importlib.util.spec_from_file_location('tensorflowjs.version', version_path)
        mod_version = importlib.util.module_from_spec(spec_version)
        spec_version.loader.exec_module(mod_version)
        setattr(tfjs_pkg, 'version', mod_version)
        sys_modules['tensorflowjs.version'] = mod_version

        # Preload converters/common and expose as tensorflowjs.converters.common
        converters_dir = os.path.join(base_tfjs, 'converters')
        converters_pkg = types.ModuleType('tensorflowjs.converters')
        common_path = os.path.join(converters_dir, 'common.py')
        spec_common = importlib.util.spec_from_file_location('tensorflowjs.converters.common', common_path)
        mod_common = importlib.util.module_from_spec(spec_common)
        spec_common.loader.exec_module(mod_common)
        setattr(converters_pkg, 'common', mod_common)
        sys_modules['tensorflowjs.converters'] = converters_pkg
        sys_modules['tensorflowjs.converters.common'] = mod_common
        # expose the converters package on the fake tensorflowjs package
        setattr(tfjs_pkg, 'converters', converters_pkg)

        # Now load keras_h5_conversion directly (it will import from our fake
        # package instead of executing the real package __init__)
        spec = importlib.util.spec_from_file_location('keras_h5_conversion_local', kh5_path)
        kh5 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kh5)

        # Convert the HDF5 Keras model to TF.js artifacts
        topology, weight_groups = kh5.h5_merged_saved_model_to_tfjs_format(
            'oral_cancer_model.h5', split_by_layer=False)
        kh5.write_artifacts(topology, weight_groups, 'tfjs_model')
        print('Conversion completed: tfjs_model/ created')
        return 0
    except Exception as e:
        print('Conversion failed:', e)
        raise

if __name__ == '__main__':
    raise SystemExit(main())
